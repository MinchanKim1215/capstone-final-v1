"""Multi-label packet/physical evidence, clipped at250ms, never predictor input."""
from collections import Counter
import numpy as np
from . import ARTIFACTS,PARENT,WORKSPACE
from .contracts import read,write,sha,now,milestone,status
from .state import from_replay
from forecast_rebuild.targets import onsets
from forecast_recall.episodes import cohort
from forecast_recall.diagnose import starts
from forecast_rebuild.calibration import transform
from forecast_temporal.inference import parent_entry

RULES=dict(version='multi-evidence-v1',packet_window='Births in(event-1.25,event-.25], exactly matching settled1s timelyPDR at onset.',
    bad_packet='Not delivered or delay>.25s; all service events retained.',
    packet_evidence='Only attempts/no-route opportunities BEFORE own deadline. Wait components intersect[born,born+.25].',
    queue_wait='At least25%ofbadpackets spent>=.125s waiting before hop first transmissions withindeadline.',
    service_wait='At least25%ofbadpackets spent>=.125s in attempted-hop service/retries/scheduling withindeadline.',
    physical_window='Actual channel intervals[event-1.25,event); component removal is a frozen instantaneous connectivity diagnostic, not a full causal mission re-simulation.',
    causes_overlap=True,exclusive_true_root_cause_claim=False,predictor_inputs_include_truth=False)

def register():
    p=ARTIFACTS/'cause_rulebook.json'
    if p.exists() and read(p)!=RULES:raise ValueError('Cause rules changed')
    if not p.exists():write(p,RULES)

def annotate(a,when):
    born=a['packet_born'];status_code=a['packet_status'];delay=a['packet_delay'];take=(born>when-1.25)&(born<=when-.25)
    bad=take&~((status_code==1)&(delay<=.25+1e-10));ids=np.flatnonzero(bad);count=len(ids)
    if count==0:raise ValueError('Service event has no bad settled-window packets')
    if (a['packet_end'][ids]<0).any():raise ValueError('Eligible event touches uninstrumented mission tail')
    ev=a['packet_evidence_before_deadline'][ids];qw=a['packet_queue_before_deadline'][ids];sw=a['packet_service_before_deadline'][ids]
    t=np.arange(len(a['physical_truth']))*.5;ph=a['physical_truth'][(t>=when-1.25)&(t<when)]
    disconnected=ph[:,0]<.5
    flags=dict(queue_wait_evidence=float((qw>=.125-1e-10).mean())>=.25,
        attempted_hop_service_evidence=float((sw>=.125-1e-10).mean())>=.25,
        no_route_head_evidence=bool((ev[:,0]>0).any()),retry_failure_evidence=bool((ev[:,1]>0).any()),
        same_draw_interference_failure_evidence=bool((ev[:,2]>0).any()),obstructed_failed_attempt_evidence=bool((ev[:,3]>0).any()),
        physical_disconnection_evidence=bool(disconnected.any()),
        interference_connectivity_repair=bool((disconnected&(ph[:,1]>.5)).any()),
        obstruction_connectivity_repair=bool((disconnected&(ph[:,2]>.5)).any()),
        fading_connectivity_repair=bool((disconnected&(ph[:,3]>.5)).any()),
        jam_started_after_event_minus3s=bool(when-3<a['jam_start']<=when),
        overflow_drop_evidence=bool((status_code[ids]==4).any()),TTL_drop_evidence=bool((status_code[ids]==2).any()),retry_limit_drop_evidence=bool((status_code[ids]==3).any()),late_delivery_evidence=bool((status_code[ids]==1).any()))
    return dict(flags=flags,bad_packets=count,total_settled_packets=int(take.sum()),status_counts={str(k):int((status_code[ids]==k).sum()) for k in range(1,5)},
        mean_queue_before_deadline=float(qw.mean()),mean_service_before_deadline=float(sw.mean()),bad_packet_ids=ids.tolist())

def event_annotations(item):
    parent=item['parent_item'];ep=cohort(parent);duration=len(ep.links)*.5
    es=np.flatnonzero(onsets(ep.links)[:,1,0])*.5;es=es[(es>=24)&(es<=duration-8.5)]
    with np.load(item['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    return [(float(when),annotate(a,float(when))) for when in es]

def main():
    register();items=read(ARTIFACTS/'selection_replay_manifest.json')['records'];entry=parent_entry();summary=read(entry['selection_summary']);policy=read(entry['policy']);threshold=read(WORKSPACE/'forecast_current/active_profile.json')['threshold'];lookup={c['key']:c for c in summary['cases']}
    cases=[];events=[];reasons=Counter();flags=Counter();miss_flags=Counter();precursors={str(o):[] for o in (-10,-7,-5,-3,-1,0)}
    status('stage1_event_cause_analysis',complete=0,total=len(items),next_action='200개 사건과134개 위험 점수 미달 사건의 증거를 분류하고1단계 완료 근거를 저장한다.')
    for index,item in enumerate(items):
        parent=item['parent_item'];ep=cohort(parent);c=lookup[item['key']]
        with np.load(c['predictions_path']) as z:score=transform(z['prediction'],policy['parameters'])[...,5]
        aa=np.array([t for t,a in starts(ep,score,ep.legacy_support,threshold) if a[0,0]]);times=ep.ends*.5;network,known=from_replay(item['path'])
        for when,outcome in event_annotations(item):
            pre=(times>when-7)&(times<=when-3)&(times>=21);valid=pre&ep.legacy_support[:,0,0]&np.isfinite(score[:,0,0]);hit=bool(((aa>when-7)&(aa<=when-3)).any());high=bool((score[valid,0,0]>=threshold).any())
            reason='timely_first_alert' if hit else 'no_qualified_input' if not valid.any() else 'risk_below_threshold' if not high else 'already_active_without_new_alert'
            reasons[reason]+=1
            for name,value in outcome['flags'].items():
                flags[name]+=int(value)
                if reason=='risk_below_threshold':miss_flags[name]+=int(value)
            align={}
            for offset in (-10,-7,-5,-3,-1,0):
                q=np.searchsorted(times,when+offset,side='right')-1
                if q<0 or when+offset-times[q]>1.01:continue
                frame=int(ep.ends[q])-1;v=network[frame];m=known[frame]
                row=dict(reported_overdue=bool((v[m,2]>0).any()),reported_no_route=bool((v[m,5]>0).any()),known_node_fraction=float(m.mean()),
                    oldest_reported_age=float(v[m,0].max()) if m.any() else None,risk=float(score[q,0,0]),query_time=float(times[q]))
                align[str(offset)]=row;precursors[str(offset)].append(row)
            events.append(dict(key=item['key'],region=item['region'],kind=item['source_record']['spec']['kind'],event_time=when,forecast_outcome=reason,**outcome,observed_precursors=align))
        cases.append(dict(key=item['key'],path=item['path'],sha256=item['sha256'],prediction_sha256=c['prediction_sha256']));status('stage1_event_cause_analysis',complete=index+1,total=len(items))
    expected=dict(timely_first_alert=60,risk_below_threshold=134,already_active_without_new_alert=5,no_qualified_input=1)
    if dict(reasons)!=expected:raise ValueError('Original200event/134miss cohort changed: '+str(reasons))
    timeline={}
    for offset,rows in precursors.items():
        timeline[offset]=dict(events=len(rows),fraction_reported_overdue=float(np.mean([r['reported_overdue'] for r in rows])),fraction_reported_no_route=float(np.mean([r['reported_no_route'] for r in rows])),
            median_known_node_fraction=float(np.median([r['known_node_fraction'] for r in rows])),median_max_reported_queue_age=float(np.median([r['oldest_reported_age'] for r in rows])),median_risk=float(np.median([r['risk'] for r in rows])))
    value=dict(created_at=now(),scope='Selection45 only',event_count=len(events),forecast_outcomes=dict(reasons),all_event_evidence_counts=dict(flags),low_score134_evidence_counts=dict(miss_flags),
        causes_overlap=True,component_removals_are_instantaneous_evidence_not_full_mission_causal_proof=True,diagnostic_truth_never_predictor_input=True,rulebook_sha256=sha(ARTIFACTS/'cause_rulebook.json'),
        observed_precursor_timeline=timeline,cases=cases,events=events,original_final_test_opened=False)
    write(ARTIFACTS/'selection_event_causes.json',value)
    names={'queue_wait_evidence':'마감 전 대기시간125ms 이상 증거','attempted_hop_service_evidence':'전송·재시도·슬롯 대기125ms 이상 증거','no_route_head_evidence':'마감 전 경로 부재로 전송 못 한 증거','retry_failure_evidence':'마감 전 전송 실패 증거',
        'same_draw_interference_failure_evidence':'동일 전송 난수에서 간섭 제거 시 성공 가능','obstructed_failed_attempt_evidence':'차폐된 링크에서 실패한 증거','physical_disconnection_evidence':'물리 연결 단절 동반','interference_connectivity_repair':'간섭 제거 시 순간 연결 회복',
        'obstruction_connectivity_repair':'건물 추가손실 제거 시 순간 연결 회복','fading_connectivity_repair':'페이딩 제거 시 순간 연결 회복','jam_started_after_event_minus3s':'사건3초 전 이후 간섭 시작','overflow_drop_evidence':'큐 초과 폐기 동반','TTL_drop_evidence':'수명 초과 폐기 동반','retry_limit_drop_evidence':'재전송 한도 폐기 동반','late_delivery_evidence':'250ms 초과 도착 동반'}
    lines=['# 캡스톤 디자인 II — 1단계 장애 원인·전조 분석','',f'선택45비행의 사건200건. 제때 경보60건, 위험 점수 미달134건, 이미 켜진 경보5건, 입력 자격 없음1건. 기존 사건 집계와 정확히 일치.','',
        '아래 항목은 중복 가능한 증거다. 합계가200건 또는134건이 될 필요가 없다. 건물·간섭 제거는 해당 순간 연결성의 대조 계산이며 전체 임무의 단일 원인 확정이 아니다.','',
        '| 관측·진단 증거 | 전체200건 | 위험 점수 미달134건 |','|---|---:|---:|']
    for name,label in names.items():lines.append(f'| {label} | {flags[name]} | {miss_flags[name]} |')
    lines+=['','## 예측 시점에 실제 받은 정보의 전조','', '| 사건 전 시점 | 마감 초과 대기 보고 비율 | 경로 부재 보고 비율 | 알려진 기체 비율 중앙값 | 위험도 중앙값 |','|---|---:|---:|---:|---:|']
    for offset,row in timeline.items():lines.append(f'| {offset}초 | {row["fraction_reported_overdue"]*100:.1f}% | {row["fraction_reported_no_route"]*100:.1f}% | {row["median_known_node_fraction"]*100:.1f}% | {row["median_risk"]:.3f} |')
    lines+=['','전조 비율은 새 관측의 시간적 분포를 설명한다. 전조가 있는 정상 시점의 빈도와 구별 능력은 다음 단계의 대조 학습에서 확인한다. 이 표만으로 예측 가능성의 상한을 단정하지 않는다.','',
        '패킷 대기·전송 시간과 실패 증거는 각 패킷의250ms 마감 전으로 제한했다. 마감 뒤의 활동을 당시 실패 원인으로 혼합하지 않았다. 원래 궤적·RF·비컨 수신·패킷 정답은 동일하다.','',
        '다음 단계: 시간 제한·수신 여부·비컨 길이를 검사한 새 관측 입력을 훈련245·선택45·보정45비행에 구성하고, 기존 입력 대조군과 비교한다.','']
    (ARTIFACTS.parent/'STAGE1_CAUSES.md').write_text('\n'.join(lines),encoding='utf-8');milestone(1,'STAGE1_CAUSES.md 및 artifacts/selection_event_causes.json; 기존200건·134건 집계 일치','훈련245·선택45·보정45비행의 새 관측을 재구성하고 실제 수신만으로 입력을 구성한다.');print(dict(reasons),dict(miss_flags),flush=True)

if __name__=='__main__':main()
