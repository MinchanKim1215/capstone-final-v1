"""Verify the compact traces and summarize the map-by-map causal audit."""
from collections import Counter

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import KINDS, LEGACY, maps, now, read, sha, write
from .radio import minimal_masks, names
from forecast_prevention.diagnostics import connected, outage_runs, reachability


RF_LABEL = dict(clear='무간섭', intermittent='간헐 간섭', continuous='지속 간섭')
TERM_LABEL = dict(building='건물 추가손실', interference='외부 간섭', shadow='섀도잉',
                  building_change='건물 추가손실 변화', distance_change='거리 감쇠 변화',
                  interference_change='수신점 간섭 변화', shadow_change='섀도잉 변화')
PHASE_LABEL = ('계획 경로 수색', '잔여 셀 수색', '수색 후 정보 공유')
ROLE_LABEL = dict(train='훈련 지도', validation='개발 검증 지도', historical_validation='과거 검증 지도',
                  test='이전 최종 평가 지도', quarantine='지리 중복 격리 지도')
PACKET_FLAGS = dict(no_route_head_evidence='마감 전 전달 경로 부재', queue_wait_evidence='마감 전 125ms 이상 대기 증거',
                    attempted_hop_service_evidence='마감 전 125ms 이상 전송·서비스 증거',
                    retry_failure_evidence='마감 전 전송 시도 실패', obstructed_failed_attempt_evidence='차폐 링크 전송 실패',
                    same_draw_interference_failure_evidence='같은 전송 난수에서 간섭 제거 시 성공',
                    overflow_drop_evidence='큐 초과 폐기', TTL_drop_evidence='수명 초과 폐기',
                    retry_limit_drop_evidence='재시도 한도 초과 폐기', late_delivery_evidence='기한 뒤 도착')


def verified_row(record, case):
    if sha(record['path']) != record['sha256']:
        raise ValueError('Map JSON changed after native simulation')
    row = read(record['path'])
    if row['contract']['case'] != case or sha(row['archive_path']) != row['archive_sha256']:
        raise ValueError('Map or trace contract changed')
    if not all((row['audited']['passed'], row['native_radio_exact'], row['native_packet_exact'])):
        raise ValueError('Native trace was not exact or safe')
    with np.load(row['archive_path'], allow_pickle=False) as trace:
        frames = int(trace['frames'])
        if frames*.5 != row['mission']['duration_s']:
            raise ValueError('Mission and trace times disagree')
        links = trace['actual_links']
        actual = connected(links, row['mission']['search_drones'])
        phase = trace['phases']
        row['phase_seconds'] = (np.bincount(phase, minlength=3)*3).tolist()
        last_health = None
        for kind in KINDS:
            entry = row['rf_cases'][kind]
            repair = np.unpackbits(trace['healthy_'+kind], axis=-1, count=frames).astype(bool)
            if last_health is not None and (repair[0] & ~last_health).any():
                raise ValueError('More interference unexpectedly improved physical connectivity on the same trajectory')
            last_health = repair[0]
            service = np.unpackbits(trace['service_'+kind], count=frames).astype(bool)
            if kind == row['spec']['kind']:
                np.testing.assert_array_equal(actual, repair[0])
            intervals = outage_runs(repair[0])
            events = entry['network_events']
            if len(intervals) != len(events) or len(outage_runs(service)) != entry['packet']['service_runs']:
                raise ValueError('Network or service tally is inconsistent')
            for (start,stop), event in zip(intervals,events):
                if event['onset_s'] != start*.5 or event['duration_s'] != (stop-start)*.5:
                    raise ValueError('Network cut interval is inconsistent')
                for end, label in [(start+2,'first_second_minimal_repairs'), (stop,'entire_cut_minimal_repairs')]:
                    viable = np.flatnonzero(repair[1:,start:end].all(axis=1))+1
                    if event[label] != [names(k) for k in minimal_masks(viable)]:
                        raise ValueError('Reported attribution differs from the saved RF counterfactual')
        row['beacon_gap_audit'] = beacon_gap_audit(native(row)['network_events'], links, row['mission']['search_drones'])
    return row


def native(row):
    return row['rf_cases'][row['spec']['kind']]


def beacon_gap_audit(events, links, relay):
    result = Counter()
    for event in events:
        obs = event['observed_before']
        ticks = obs['decision_window']
        if event['left_censored'] or not ticks or obs['qualified_decisions']:
            continue
        indirect_candidate = False
        any_all_reachable = False
        for tick in ticks:
            missing = tick['missing_search_beacons']
            if not missing:
                continue
            graph = links[int(tick['time_s']*2)-1]
            direct = graph[relay, missing]
            reachable = reachability(graph)[relay, missing]
            indirect = reachable & ~direct
            result['missing_peers_observations'] += len(missing)
            result['direct_up'] += int(direct.sum())
            result['indirect_only'] += int(indirect.sum())
            result['unreachable'] += int((~reachable).sum())
            any_all_reachable |= bool(reachable.all())
            indirect_candidate |= bool(reachable.all() and indirect.any() and tick['local_recent_rf_links'])
        result['event_with_all_missing_reachable_on_a_tick'] += any_all_reachable
        result['event_with_route_for_indirect_forwarding'] += indirect_candidate
    return result


def metrics(rows, kind=None):
    records = [native(row) if kind is None else row['rf_cases'][kind] for row in rows]
    hours = sum(row['mission']['duration_s'] for row in rows)/3600
    events = [event for row in records for event in row['network_events']]
    new = sum(not e['left_censored'] for e in events)
    duration = sum(e['duration_s'] for e in events)
    generated = sum(row['packet']['generated_packets'] for row in records)
    timely = sum(row['packet']['timely_pdr']*row['packet']['generated_packets'] for row in records)
    return dict(maps=len(rows), hours=hours, events=len(events), onsets=new,
                events_per_h=new/hours, cut_s=duration, cut_s_per_h=duration/hours,
                worst_cut_s=max((e['duration_s'] for e in events), default=0.),
                cut_s_p50_p90=np.quantile([e['duration_s'] for e in events], [.5,.9]).tolist() if events else [0.,0.],
                timely_pdr=timely/max(1,generated), service_runs=sum(r['packet']['service_runs'] for r in records),
                undirected_link_runs=sum(r['individual_link_runs'] for r in records),
                mission_completed=sum(r['mission']['mission_completed'] for r in rows),
                generated_packets=generated, maps_with_cuts=sum(bool(r['network_events']) for r in records))


def causes(events):
    events = list(events)
    first, full, trigger, topology, observations = (Counter() for _ in range(5))
    for event in events:
        for key, counter in [('first_second_minimal_repairs',first), ('entire_cut_minimal_repairs',full),
                             ('minimal_prior_term_restorations',trigger)]:
            if key in event:
                parts = event[key]
                for values in parts:
                    counter['+'.join(values)] += 1
                counter['combined_only'] += bool(parts) and all(len(r)>1 for r in parts)
                counter['unresolved'] += not parts
        topology['isolated_search'] += bool(event['isolated_search_nodes'])
        topology['three_or_more_search_components'] += event['search_components'] >= 3
        topology['single_new_edge_can_repair'] += any(edge['reconnects_search_alone'] for edge in
                                                    event.get('newly_failed_cross_component_edges', []))
        edge_roles = {edge['role_pair'] for edge in event.get('newly_failed_cross_component_edges', [])
                      if edge['reconnects_search_alone']}
        topology.update('one_edge_'+role for role in edge_roles)
        if not event['left_censored'] and 'observed_before' in event:
            obs = event['observed_before']
            window = obs['decision_window']
            if not window:
                observations['no_decision'] += 1
            elif obs['qualified_decisions']:
                observations['any_qualified'] += 1
            else:
                observations['no_qualified'] += 1
                observations['all_ticks_missing_search'] += all(t['missing_search_beacons'] for t in window)
                observations['all_ticks_missing_local_rf'] += all(t['local_recent_rf_links']==0 for t in window)
                observations['any_never_received_search'] += any(t['never_received_search_beacons'] for t in window)
    return dict(first_second=first, whole_cut=full, transition=trigger, topology=topology,
                observations=observations, residual_cut_seconds=sum(e['still_down_without_all_three_seconds'] for e in events),
                recovered_cut_seconds={factor:sum(e['duration_s']*e['restored_fraction_by_removal'][1 << i]
                                                 for e in events)
                                       for i,factor in enumerate(('building','interference','shadow'))})


def service(rows):
    events = [e for row in rows for e in native(row)['service_events']]
    categories, flags, healthy_flags = Counter(), Counter(), Counter()
    for e in events:
        categories['network_down_at_onset'] += e['network_down_at_onset']
        categories['network_down_before_or_during'] += (e['network_down_any_during_event'] or
                                                       e['network_down_during_settled_birth_window'])
        whole_healthy = not (e['network_down_any_during_event'] or e['network_down_during_settled_birth_window'])
        categories['network_up_before_and_during'] += whole_healthy
        categories['attributed'] += 'flags' in e
        if 'flags' in e:
            categories['attributed_with_network_up'] += whole_healthy
            flags.update({key:int(v) for key,v in e['flags'].items()})
            if whole_healthy:
                healthy_flags.update({key:int(v) for key,v in e['flags'].items()})
    return dict(events=len(events), categories=categories, flags=flags, network_up_flags=healthy_flags)


def rate_line(title, r):
    return (f'| {title} | {r["maps"]} | {r["hours"]:.2f} | {r["events"]:,} | {r["onsets"]:,} | '
            f'{r["events_per_h"]:.2f} | {r["cut_s_per_h"]:.2f} | {r["timely_pdr"]:.2%} | {r["service_runs"]:,} |')


def top_examples(rows):
    events = [(row, event) for row in rows for event in native(row)['network_events'] if not event['left_censored']]
    first = lambda event: {tuple(r) for r in event['first_second_minimal_repairs']}
    groups = [('건물 중심', [(r,e) for r,e in events if first(e)=={('building',)}]),
              ('간섭 중심', [(r,e) for r,e in events if first(e)=={('interference',)}]),
              ('여러 요인 결합', [(r,e) for r,e in events if first(e) and all(len(x)>1 for x in first(e))])]
    output = []
    for name, group in groups:
        if group:
            row, event = max(group, key=lambda item: (item[1]['duration_s'], -item[1]['onset_s']))
            output.append((name, row, event))
    return output


def roofs_on_edge(entry, event, edge):
    from shapely.geometry import LineString, shape
    from swarm_sim.regions import read_json
    scene = read_json(LEGACY/entry['path'])
    i,j = edge['nodes']
    segment = LineString([event['positions_xy_m'][i],event['positions_xy_m'][j]])
    roofs = []
    for building in scene['buildings']:
        if building['height_m'] < 40:
            continue
        section = segment.intersection(shape(building['geometry_local_m']))
        if section.length > 0:
            roofs.append((building['height_m'], building['building_id'], building['height_source']))
    return sorted(roofs, key=lambda item: -item[0])


def finish(plan, records):
    expected = {c['spec']['region']: c for c in plan['cases']}
    if len(records) != len(expected) or {r['region'] for r in records} != set(expected):
        raise ValueError('Full map coverage failed')
    rows = [verified_row(record, expected[record['region']]) for record in records]
    registry = maps()
    native_metrics = metrics(rows)
    rf_metrics = {kind:metrics(rows,kind) for kind in KINDS}
    native_events = [e for row in rows for e in native(row)['network_events']]
    native_cause = causes(native_events)
    by_kind = {kind:causes(e for row in rows for e in row['rf_cases'][kind]['network_events']) for kind in KINDS}
    phases = [sum(row['phase_seconds'][i] for row in rows) for i in range(3)]
    phase_counts = Counter(e['phase'] for e in native_events if not e['left_censored'])
    packet = service(rows)
    jammer_change = Counter()
    for row in rows:
        for event in native(row)['network_events']:
            if ['interference_change'] in event.get('minimal_prior_term_restorations', []):
                t = event['onset_s']
                just_started = row['spec']['kind']=='intermittent' and t-.5 < row['jammer']['intermittent_start_s'] <= t
                jammer_change['start' if just_started else 'already_active'] += 1
    split_metrics = {split:metrics([r for r in rows if r['contract']['case']['split']==split]) for split in ROLE_LABEL}
    extra_interference = {kind:sum(r['rf_cases'][kind]['disconnected_seconds']>
                                  r['rf_cases']['clear']['disconnected_seconds'] for r in rows) for kind in KINDS[1:]}
    report = ['# 캡스톤 디자인 II — 전체 지도 통신 단절 원인 조사', '',
              f'등록 지도 **{len(rows)}개 전체**에서 V7 PPO로 새 비행을 수행했다. '
              f'이 중 {native_metrics["mission_completed"]}개가 수색 임무를 완주했다. '
              f'같은 궤적을 무간섭·간헐 간섭·지속 간섭으로 재생한 결과는 **{len(rows)*3:,}조건**이다. '
              '지도마다 비행 방향과 난수 한 세트를 사전에 배정했다.', '',
              '## 전체 결과', '',
              '| 비행·재생 조건 | 지도 | 총 비행 시간(h) | 지속 단절 구간 | 새 단절 | 새 단절/시간 | 단절 초/시간 | 기한 내 전달 | 서비스 장애 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|', rate_line('실제 제어 비행',native_metrics)]
    report += [rate_line(RF_LABEL[k],rf_metrics[k]) for k in KINDS]
    report += ['', '수색기들이 직접 또는 중계 경로로 1초 이상 연결되지 않으면 지속 단절로 기록했다. '
               '이미 끊긴 채 시작한 구간은 전체 구간에는 포함하고 새 단절에서는 제외했다. '
               f'실제 제어 비행에서 수색망 지속 단절을 경험한 지도는 {native_metrics["maps_with_cuts"]}개다. '
               f'개별 링크 단절은 {native_metrics["undirected_link_runs"]:,}구간이었고, '
               f'수색망이 가장 오래 끊긴 한 구간은 {native_metrics["worst_cut_s"]:.1f}초였다. '
               f'수색망 단절 시간의 중앙값은 {native_metrics["cut_s_p50_p90"][0]:.1f}초, '
               f'90백분위는 {native_metrics["cut_s_p50_p90"][1]:.1f}초다.', '',
               '조건별 재생에서는 지도별로 실제 비행에서 얻은 동일 궤적과 같은 섀도잉을 사용했다. '
               '간섭을 켠 조건끼리는 간섭원의 위치와 송신 전력을 맞췄다. 패킷 전송·재시도는 각 조건에서 다시 계산했다. '
               '다른 간섭에서 PPO가 새로 고를 궤적은 지도별 제어 비행 한 번의 결과에 포함되지 않는다.', '',
               f'동일 궤적에서 간섭으로 연결 상실 시간이 늘어난 지도는 간헐 간섭 {extra_interference["intermittent"]}개, '
               f'지속 간섭 {extra_interference["continuous"]}개였다. 이 지도 수에는 0.5초짜리 연결 상실도 포함했다.', '',
               '## 전파 단절 원인과 시작 순간', '',
               '| 각각 제거한 요인 | 제어 비행에서 첫 1초 전부 복구 | 제어 비행에서 단절 전체 복구 | 무간섭 첫 1초 | 간헐 간섭 첫 1초 | 지속 간섭 첫 1초 |',
               '|---|---:|---:|---:|---:|---:|']
    for name in ('building','interference','shadow','combined_only','unresolved'):
        label = TERM_LABEL.get(name, '두 요인 이상 함께 제거해야 복구' if name=='combined_only' else '세 요인을 모두 제거해도 구간 전체 미복구')
        values = [native_cause['first_second'][name],native_cause['whole_cut'][name]]
        values.extend(by_kind[k]['first_second'][name] for k in KINDS)
        report.append('| ' + ' | '.join([label,*(f'{n:,}' for n in values)]) + ' |')
    report += ['', '단절 구간의 모든 0.5초 프레임에서 건물 추가손실·외부 간섭·섀도잉을 유지하거나 제거하는 8개 조합을 비교했다. '
               '한 사건을 둘 이상의 단독 제거가 복구할 수 있으므로 원인별 행은 겹칠 수 있다. '
               '거리는 그대로 유지했다. 세 항을 모두 제거한 뒤에도 실제 단절 구간 안에서 연결되지 않은 시간은 '
               f'총 {native_cause["residual_cut_seconds"]:.1f}초였다.', '',
               '새 단절 시작에서는 직전 0.5초 값으로 개별 항을 되돌려 어떤 변화가 연결 상실에 기여했는지 추가로 확인했다.', '',
               '| 직전 값으로 되돌렸을 때 시작 프레임을 복구한 항 | 제어 비행 사건 |', '|---|---:|']
    for name in ('building_change','distance_change','interference_change','shadow_change','combined_only'):
        report.append(f'| {TERM_LABEL.get(name, "두 항 이상 함께 되돌림")} | {native_cause["transition"][name]:,} |')
    fixed = native_cause['recovered_cut_seconds']
    report += ['', f'지속 단절 시간 중 단독 제거로 회복한 시간은 건물 {fixed["building"]:,.1f}초, '
               f'간섭 {fixed["interference"]:,.1f}초, 섀도잉 {fixed["shadow"]:,.1f}초다. '
               f'직전 간섭 값만 되돌려 시작 프레임이 복구된 경우 중 간섭원이 그때 켜진 사건은 {jammer_change["start"]}건, '
               f'이미 켜져 있던 사건은 {jammer_change["already_active"]}건이었다.', '']
    top = native_cause['topology']
    report += ['', f'수색기 한 대 이상이 단독 성분으로 고립된 구간은 {top["isolated_search"]:,}개였다. '
               f'수색망이 세 부분 이상으로 나뉜 구간은 {top["three_or_more_search_components"]:,}개였다. '
               f'새로 끊긴 링크 하나를 복구하면 수색망 전체가 다시 이어지는 시작은 {top["single_new_edge_can_repair"]:,}건이었다. '
               f'그중 수색기–중계기 링크가 복구 후보인 사건은 {top["one_edge_search-relay"]:,}건이다. '
               '이는 단절 프레임에서의 그래프 대조다. 어떤 이동으로 그 링크를 살릴 수 있는지는 별도의 예방 제어 시험에서 판정한다.', '',
               '| 임무 단계 | 비행 시간(h) | 해당 단계에서 시작한 단절 | 시간당 시작 |', '|---|---:|---:|---:|']
    for i, phase in enumerate(PHASE_LABEL):
        report.append(f'| {phase} | {phases[i]/3600:.2f} | {phase_counts[i]:,} | {3600*phase_counts[i]/max(1,phases[i]):.2f} |')
    obs = native_cause['observations']
    beacon_gaps = sum((row['beacon_gap_audit'] for row in rows), Counter())
    report += ['', '## 단절을 앞두고 실제로 받은 관측', '',
               f'3초 이상 7초 미만 선행 구간에서 첫 중계기가 수색기 경로 비컨과 최근 RF 수신을 모두 확보한 결정 시점이 있었던 사건은 '
               f'{obs["any_qualified"]:,}건이다. 적합한 시점이 없었던 사건은 {obs["no_qualified"]:,}건, '
               f'시작 21초 이후의 3초 제어 시점 자체가 없었던 사건은 {obs["no_decision"]:,}건이다. '
               '이는 예측에 필요한 자료를 받을 수 있었는지의 검사다.', '',
               f'수신 요건을 충족하지 못한 사건 중 모든 평가 시점에서 수색기 비컨이 누락되거나 3초보다 오래된 경우는 '
               f'{obs["all_ticks_missing_search"]:,}건, 첫 중계기에 최근 RF가 전혀 없었던 경우는 {obs["all_ticks_missing_local_rf"]:,}건이다. '
               f'적어도 한 시점에 수색기 비컨을 한 번도 받지 못한 기체가 있었던 사건은 {obs["any_never_received_search"]:,}건이다. '
               '세 집계에는 같은 사건이 들어갈 수 있다.', '',
               '최근 비컨이 부족했던 수색기와 시점을 따로 세었다. '
               f'결정 직전의 실제 데이터 링크 기준으로 첫 중계기와 직접 연결 {beacon_gaps["direct_up"]:,}회, '
               f'다른 기체를 거쳐 연결 {beacon_gaps["indirect_only"]:,}회, 경로 없음 {beacon_gaps["unreachable"]:,}회였다. '
               f'{beacon_gaps["event_with_route_for_indirect_forwarding"]:,}건은 최근 RF가 있는 결정 시점에 '
               '누락된 비컨을 모두 전달할 경로가 있고 그중 하나 이상이 우회를 필요로 했다. '
               '이는 진단용 그래프 검사다. 실제 비컨 전달 지연·성공률은 따로 검증해야 한다.', '',
               '## 패킷 서비스 장애', '']
    cat = packet['categories']
    report += [f'250ms 기한 내 전달률이 최근 1초 동안 90% 미만인 상태가 1초 이상 이어진 장애는 {packet["events"]:,}구간이다. '
               f'시작 순간 수색망 단절이 동반된 구간은 {cat["network_down_at_onset"]:,}개다. '
               f'직전 패킷 판정 구간과 장애 전체에서 수색망이 연결됐던 구간도 {cat["network_up_before_and_during"]:,}개였다.', '',
               f'마감 전 패킷 기록을 추적할 수 있는 {cat["attributed"]:,}건을 집계했다. '
               f'그중 망이 연결돼 있던 {cat["attributed_with_network_up"]:,}건을 별도로 표시했다. '
               '같은 사건에서 여러 패킷 증거가 함께 나타날 수 있다.', '',
               '| 마감 전 패킷 증거 | 추적 가능 전체 | 수색망 연결 유지 사건 |', '|---|---:|---:|']
    for name, label in PACKET_FLAGS.items():
        report.append(f'| {label} | {packet["flags"][name]:,} | {packet["network_up_flags"][name]:,} |')
    report += ['', '대기열 증가는 전달 경로가 먼저 사라진 뒤 생길 수 있다. 경로 부재와 실제 전송 시도 실패를 분리해 해석했다.', '',
               '## 개별 단절 사례', '']
    for label,row,event in top_examples(rows):
        region = row['spec']['region']
        report += [f'**{label}** · {registry[region]["name"]} · {RF_LABEL[row["spec"]["kind"]]}', '',
                   f'{event["onset_s"]:.1f}초에 단절이 시작해 {event["duration_s"]:.1f}초간 지속됐다. '
                   f'시작 시 연결 성분은 `{event["components"]}`이다. 기체 인덱스는 0부터 센다. '
                   f'첫 1초를 복구한 최소 제거 조합은 `{event["first_second_minimal_repairs"]}`, '
                   f'시작 프레임을 복구한 직전 항은 `{event.get("minimal_prior_term_restorations", [])}`이다.']
        edges = event.get('newly_failed_cross_component_edges', [])
        if edges:
            edge = next((e for e in edges if e['reconnects_search_alone']), edges[0])
            before, after = edge['minimum_directional_hop_probability']
            report += ['', f'새로 끊긴 링크 `{edge["nodes"]}`의 양방향 중 낮은 홉 성공률은 {before:.2%}에서 {after:.2%}로 바뀌었다. '
                       f'같은 두 시점의 거리 {edge["distance_m"][0]:.1f}→{edge["distance_m"][1]:.1f}m, '
                       f'건물 추가손실 {edge["building_loss_db"][0]:.2f}→{edge["building_loss_db"][1]:.2f}dB였다.']
            roofs = roofs_on_edge(registry[region],event,edge) if edge['building_loss_db'][1]>0 else []
            if roofs:
                roof_list = ', '.join(f'`{key}` {h:.1f}m'+(' 추정 높이' if source=='imputed' else '')
                                      for h,key,source in roofs[:4])
                report += ['', f'이 링크의 평면 선분과 겹치는 40m 이상 건물 후보는 {len(roofs)}개다. '
                           f'높이가 큰 후보는 {roof_list}이다. 개별 건물을 제거하는 대조는 수행하지 않았다.']
        report += ['', f'[해당 지도의 세 조건과 사건별 원시 근거](artifacts/maps/{region}.json)', '']
    report += ['## 지도 집계와 해석 범위', '',
               '| 이전 구분 | 지도 | 전체 제어 비행 시간(h) | 지속 단절 구간 | 시간당 새 단절 |', '|---|---:|---:|---:|---:|']
    for split, m in split_metrics.items():
        report.append(f'| {ROLE_LABEL[split]} | {m["maps"]} | {m["hours"]:.2f} | {m["events"]:,} | {m["events_per_h"]:.2f} |')
    report += ['', '### 실제 제어 비행에서 누적 단절이 긴 지도', '',
               '| 지도 | 실제 제어 비행 조건 | 지속 단절 구간 | 누적 단절 | 최장 단절 |', '|---|---|---:|---:|---:|']
    by_time = sorted(rows, key=lambda row: -sum(e['duration_s'] for e in native(row)['network_events']))
    for row in by_time[:10]:
        region = row['spec']['region']
        events = native(row)['network_events']
        total = sum(e['duration_s'] for e in events)
        longest = max((e['duration_s'] for e in events), default=0)
        name = registry[region]['name'].replace('|','／')
        report.append(f'| [{name}](artifacts/maps/{region}.json) | {RF_LABEL[row["spec"]["kind"]]} | '
                      f'{len(events)} | {total:.1f}초 | {longest:.1f}초 |')
    report += ['', '이전 최종 평가 32개와 지리 중복 격리 1개를 진단 범위에 포함했다. '
               '이 33개는 모델 학습·문턱 선택 자료에서 분리했다. 이전 최종 평가 지도는 이번에 결과를 열었으므로 이후 모델의 미공개 최종 평가에는 새 지도가 필요하다.', '',
               '이번 수치는 고도 50m·2.4GHz와 프로젝트의 건물 높이·모의 간섭·패킷 모델에서 얻었다. '
               '건물 높이 보간, 한 방향·한 난수, 정해진 간섭 시나리오가 결과에 영향을 준다. '
               '이전 예방 제어 비교와 구분해 V7의 원인 조사 결과로 해석한다.', '',
               '[전체 345개 지도별 결과와 사건 목록](MAPS.md) · [집계 원시 자료](artifacts/summary.json)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    index = ['# 전체 지도별 통신 단절 결과', '',
             '각 지도 링크에서 새 단절 시각, 지속 시간, 임무 단계, 기체별 위치, 끊긴 링크, 전파 항별 값, '
             '단절 첫 1초와 전체 복구 조합, 실제 비행의 선행 관측·패킷 근거를 확인할 수 있다.', '',
             '표의 건수는 처음부터 끊긴 구간을 포함한 지속 단절 구간이다. 괄호 안은 누적 지속 시간이다.', '',
             '| 지도 | 이전 구분 | 실제 제어 비행 조건 | 무간섭 | 간헐 간섭 | 지속 간섭 | 실제 비행 서비스 장애 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:r['spec']['region']):
        region = row['spec']['region']
        m = registry[region]
        fields = []
        for kind in KINDS:
            events = row['rf_cases'][kind]['network_events']
            fields.append(f'{len(events)} ({sum(e["duration_s"] for e in events):.1f}초)')
        direction = '동서' if row['spec']['direction']=='east_west' else '남북'
        title = m['name'].replace('|','／')
        index.append(f'| [{title}](artifacts/maps/{region}.json) | {ROLE_LABEL[m["joint_split"]]} | '
                     f'{RF_LABEL[row["spec"]["kind"]]} · {direction} | ' + ' | '.join(fields) +
                     f' | {len(native(row)["service_events"])} |')
    (ROOT/'MAPS.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    write(ARTIFACTS/'summary.json', dict(native=native_metrics, rf_regimes=rf_metrics, native_causes=native_cause,
          causes_by_regime=by_kind, native_service=packet, native_phase_seconds=phases, native_phase_onsets=phase_counts,
          jammer_activation_at_interference_trigger=jammer_change,
          beacon_gap_graph_audit=beacon_gaps,
          maps_with_added_disconnection_from_interference=extra_interference,
          previous_split=split_metrics, maps_verified=len(rows), rf_cases_verified=len(rows)*3,
          independent_recount_passed=True, created_at=now()))
