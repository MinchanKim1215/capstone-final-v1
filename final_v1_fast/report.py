"""Matched results; risk-conditioned branches are not operational reliability."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .protocol import ARMS,CONFIG,read,write,now

LABELS={'p3_b1':'3초 판단 · 1Hz 관측','p1_b1':'1초 판단 · 1Hz 관측',
        'p05_b1':'0.5초 판단 · 1Hz 관측','p05_b10':'0.5초 판단 · 10Hz 관측',
        'p05_b10_e02':'10Hz 관측 + 0.2초 긴급 감시','p05_b10_e01':'10Hz 관측 + 0.1초 긴급 감시',
        'p05_bhybrid':'중요 링크 두 기체만 10Hz','p05_bhybrid_e01':'중요 링크 10Hz + 0.1초 긴급 감시'}


def pool(rows):
    total=sum(r['generated_packets'] for r in rows)
    return dict(cases=len(rows),onsets=sum(r['onsets'] for r in rows),
        cut_seconds=sum(r['cut_seconds'] for r in rows),
        initial_down=sum(r['initial_cut'] for r in rows),
        seconds=sum(r['duration_s'] for r in rows),
        timely_pdr=sum(r['timely_packets'] for r in rows)/max(1,total),
        data_pdr=sum(r['delivered_packets'] for r in rows)/max(1,total),
        generated_packets=total,relay_travel_m=sum(r['relay_travel_m'] for r in rows),
        search_travel_m=sum(r['search_travel_m'] for r in rows),
        reversals=sum(r['reversals'] for r in rows),
        requests=sum(len(r['interventions']) for r in rows),
        emergency_requests=sum(r['emergency_requests'] for r in rows),
        prevention_requests=sum(r['prevention_requests'] for r in rows),
        connected_requests=sum(r['connected_requests'] for r in rows),
        airtime_fraction=float(np.mean([r['reserved_airtime_fraction'] for r in rows])),
        minimum_separation_m=min(r['minimum_separation_m'] for r in rows),
        maximum_speed_mps=max(r['maximum_speed_mps'] for r in rows),
        maximum_acceleration_mps2=max(r['maximum_acceleration_mps2'] for r in rows),
        median_trial_p95_decision_ms=float(np.median([r['decision_p95_ms'] for r in rows])),
        decision_budget_exceeded=sum(r['decisions_over_budget'] for r in rows),
        assessments=sum(r['assessments'] for r in rows),
        busy_skips=sum(r['busy_planning_skips'] for r in rows),
        safety_violations=sum(not r['safety_passed'] for r in rows))


def paired(old,new):
    original={r['case_key']:r for r in old}
    groups=defaultdict(list)
    for r in new:
        b=original[r['case_key']]
        groups[r['spec']['region']].append([r['onsets']-b['onsets'],r['cut_seconds']-b['cut_seconds'],
                                          r['timely_packets']-b['timely_packets'],b['generated_packets']])
    totals=np.array([np.sum(v,axis=0) for v in groups.values()],float)
    rng=np.random.default_rng(2026110507)
    idx=rng.integers(len(totals),size=(10000,len(totals)))
    sampled=totals[idx].sum(axis=1)
    delta=sum(r['onsets'] for r in new)-sum(r['onsets'] for r in old)
    return dict(onset_delta=delta,onset_delta_ci95=np.quantile(sampled[:,0],[.025,.975]).tolist(),
        cut_seconds_delta=sum(r['cut_seconds'] for r in new)-sum(r['cut_seconds'] for r in old),
        cut_seconds_delta_ci95=np.quantile(sampled[:,1],[.025,.975]).tolist(),
        timely_pdr_pp=(pool(new)['timely_pdr']-pool(old)['timely_pdr'])*100.,
        map_groups=len(groups),cases=len(new))


def build(plan,cases,rows,stress,checks):
    grouped={a:[r for r in rows if r['arm']==a] for a in ARMS}
    stats={a:pool(v) for a,v in grouped.items()}
    comparisons={a:paired(grouped['p3_b1'],v) for a,v in grouped.items() if a!='p3_b1'}
    stress_stats={a:pool([r for r in stress if r['arm']==a]) for a in ARMS}
    base=stats['p3_b1']
    eligible=[a for a in comparisons if stats[a]['safety_violations']==0 and
              comparisons[a]['timely_pdr_pp']>=-.25 and comparisons[a]['onset_delta_ci95'][1]<0]
    result=dict(created_at=now(),scope=plan['scope'],summary=stats,comparisons=comparisons,
        sensor_effect=paired(grouped['p05_b1'],grouped['p05_b10']),
        emergency02_effect=paired(grouped['p05_b10'],grouped['p05_b10_e02']),
        emergency01_effect=paired(grouped['p05_b10'],grouped['p05_b10_e01']),
        hybrid_effect=paired(grouped['p05_b1'],grouped['p05_bhybrid']),
        hybrid_emergency_effect=paired(grouped['p05_bhybrid'],grouped['p05_bhybrid_e01']),
        stress=stress_stats,audit=checks,eligible_for_further_full_mission_study=eligible,
        by_anchor={kind:{a:pool([r for r in v if r['anchor_kind']==kind]) for a,v in grouped.items()
                         if any(r['anchor_kind']==kind for r in v)} for kind in ('outage','quiet')},
        production_promoted=False,models_retrained=False)
    write(ARTIFACTS/'results.json',result)
    figures(result)
    note(plan,cases,result)
    return result


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    f=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':f.get_name(),'axes.unicode_minus':False,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    stats=result['summary'];keys=list(ARMS)
    labels=['3초 / 1Hz','1초 / 1Hz','0.5초 / 1Hz','0.5초 / 10Hz','+ 0.2초 긴급','+ 0.1초 긴급','링크만 10Hz','링크 + 긴급']
    colors=['#687f8e','#2b7f9e','#168472','#b28531','#b56b44','#a14d49','#3f8593','#2e616e']
    fig,axes=plt.subplots(2,2,figsize=(14.8,10))
    fig.subplots_adjust(left=.08,right=.96,top=.84,bottom=.13,wspace=.28,hspace=.58)
    fig.text(.055,.95,'Final V1 · 빠른 명령 주기 대조',fontsize=23,fontweight='bold')
    fig.text(.055,.895,f'동일 출발의 24초 분기 {stats[keys[0]]["cases"]}조건 · 운동·수신·계산·명령 지연과 통신 시간 반영',fontsize=12,color='#526575')
    fields=[('onsets','새 지속 단절(건)',lambda x:x),('cut_seconds','지속 단절 시간(초)',lambda x:x),
            ('timely_pdr','기한 내 패킷 전달률(%)',lambda x:x*100),('relay_travel_m','중계 이동량(km)',lambda x:x/1000)]
    for ax,(field,title,scale) in zip(axes.ravel(),fields):
        values=[scale(stats[a][field]) for a in keys]
        ax.bar(np.arange(len(keys)),values,color=colors,width=.68)
        for i,v in enumerate(values):
            ax.text(i,v+max(values)*.025,f'{v:.1f}' if field!='onsets' else f'{v:g}',ha='center',fontsize=10,fontweight='bold')
        ax.set_xticks(np.arange(len(keys)),labels,rotation=28,ha='right',fontsize=9.)
        ax.set_ylim(0,max(values)*1.2 if max(values)>0 else 1)
        ax.set_title(title,loc='left',pad=15,fontsize=13)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.text(.5,.035,'원본 Final V1 전체 임무와 다른 연속 이동 시험이다. 위험 구간을 포함한 조건부 결과이며 실기체 성능·전체 임무 오경보율을 뜻하지 않는다.',ha='center',fontsize=10,color='#526575')
    fig.savefig(ROOT/'빠른제어_비교.png',dpi=165)
    plt.close(fig)


def note(plan,cases,result):
    stats=result['summary'];base=stats['p3_b1']
    lines=['# Final V1 · 빠른 명령 적용 주기 실험','',f'완료 시각 {result["created_at"]}','',
        '현재 Final V1 코드·가중치를 보존하고 빠른 제어를 별도 환경에서 시험했다. '
        f'기존 V21의 42개 비행에서 {len(cases)}개 출발 조건을 만들고 여덟 구성을 비교했다. '
        '각 비행에서 서로 다른 단절 구간 최대 세 곳과, 연결이 유지된 구간 한 곳을 고정 규칙으로 골랐다. '
        '이는 위험 구간을 포함한 24초 분기 실험이며 전체 임무 완료율·실운용 오경보/h의 추정 자료가 아니다.','',
        '## 시험 조건','',
        '- 최대 속도 5m/s, 가속도 2.5m/s², 최소 분리 5m. 지도에서 안전한 간선을 따라 이동한다. '
        '간선 끝에서 정지한 뒤 다음 간선을 선택하므로 이동 도중 임의 방향으로 즉시 꺾지 않는다.',
        '- 출발 위치는 원본 비행과 같고, 초기 속도와 패킷 큐는 모든 방식에서 0이다. '
        '원본의 이후 단절을 그대로 재현하는 실험이 아니며, 새 전파와 이동으로 발생한 단절을 새로 집계한다.',
        '- 원본 항법 생성기를 출발 시점까지 정확히 재현한 뒤, 그때 발행된 보정을 유지하는 미래 계획을 만든다. '
        '이후 기록된 PPO 행동이나 실제 미래 위치는 입력하지 않는다. 공유 항법 상태와 발행 계획은 정확하다고 가정한다.',
        '- 0.1초마다 위치에 맞는 전파를 새로 계산한다. 그림자 변화는 기존 0.5초 외생 과정에 조건부인 OU bridge이며, '
        '0.5초 끝점은 보존한다. 직선 분할이나 실제 기체에서 측정한 10Hz 기록이 아니다.',
        '- 성공적으로 수신된 비컨만 사용한다. 비컨에 실리는 타 수신기 보고는 이전 수신 기록이다. '
        '센서 처리 0.1초, 계획 계산 0.2초, 기본 명령 전달 0.1초를 반영했다. '
        '계획 계산 중 겹친 요청은 건너뛰며, 명령은 최대 세 번의 순차 방송으로 전달한다.',
        '- 비컨 한 대당 5ms, 명령 전달 100ms당 세 슬롯을 데이터 채널에서 차감했다. '
        '8대 기준 점유율은 1Hz 비컨 19%, 10Hz 비컨 55%다. 고정 슬롯 모형의 가정이며 실제 장비 측정값이 아니다.',
        '- 같은 물리 위험 판단·후보·보호 기준을 여덟 구성에 적용했다. PPO나 LightGBM은 재학습하지 않았다. '
        '고주기 물리 제어의 공학 대조이며 기존 V21 정책을 그대로 고주기로 실행한 성능 수치가 아니다.','',
        '개발 시험에서 전 기체 10Hz의 통신 비용이 확인돼, 새 확인 구간의 결과를 열기 전에 두 구성을 추가 등록했다. '
        '추가 구성은 모든 기체의 1Hz 보고를 유지하고, 예측한 병목 링크 양 끝 두 기체만 10Hz로 보고한다. '
        '보고 주기 변경도 지연·손실이 있는 명령으로 전달하며, 수신하지 못한 기체는 이전 주기를 유지한다. '
        '두 기체는 전파 정답이 아닌 수신 자료와 발행 경로의 예측으로 고른다. '
        '최초 등록은 protocol.json, 추가 등록은 protocol_v2.json에 보관했다.','',
        '## 같은 출발 조건의 결과','',
        '| 구성 | 새 단절 | 단절 시간(초) | 기한 내 전달률 | 중계 이동(km) | 방향 반전 | 요청 수 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,s in stats.items():
        lines.append(f'| {LABELS[a]} | {s["onsets"]} | {s["cut_seconds"]:.1f} | {s["timely_pdr"]:.2%} | {s["relay_travel_m"]/1000:.2f} | {s["reversals"]} | {s["requests"]} |')
    lines+=['','## 3초 판단 대비 지도 단위 대조','','| 구성 | 단절 수 차이 | 지도 재표집 95% 구간 | 전달률 차이 |','|---|---:|---:|---:|']
    for a,c in result['comparisons'].items():
        lo,hi=c['onset_delta_ci95']
        lines.append(f'| {LABELS[a]} | {c["onset_delta"]:+d} | [{lo:+.1f}, {hi:+.1f}] | {c["timely_pdr_pp"]:+.2f}%p |')
    lines+=['','지도 14개를 단위로 10,000번 재표집했다. 표의 구간은 같은 크기 시험집합의 차이에 대한 '
        '불확실성이다. 지도 수와 선택된 출발 조건이 제한돼 있으며 새 지도 일반화 구간으로 해석하지 않는다.','',
        '## 추가 효과 분리','']
    for name,key in [('10Hz 관측 추가','sensor_effect'),('0.2초 긴급 감시 추가','emergency02_effect'),('0.1초 긴급 감시 추가','emergency01_effect'),('중요 링크만 10Hz','hybrid_effect'),('중요 링크 10Hz의 긴급 감시 추가','hybrid_emergency_effect')]:
        c=result[key];lo,hi=c['onset_delta_ci95']
        lines.append(f'- {name}: 새 단절 {c["onset_delta"]:+d}건, 95% 구간 [{lo:+.1f}, {hi:+.1f}], 기한 내 전달 {c["timely_pdr_pp"]:+.2f}%p.')
    lines+=['','## 명령 전달 지연 0.3초 대조','',
        '확인 결과와 무관하게 지도마다 한 조건을 고정해 추가 비교했다. 센서·계산 지연은 기본 시험과 같다.','',
        '| 구성 | 조건 수 | 새 단절 | 단절 시간(초) | 기한 내 전달률 |','|---|---:|---:|---:|---:|']
    for a,s in result['stress'].items():
        lines.append(f'| {LABELS[a]} | {s["cases"]} | {s["onsets"]} | {s["cut_seconds"]:.1f} | {s["timely_pdr"]:.2%} |')
    lines+=['','## 안전과 계산 시간','',
        f'전체 {result["audit"]["trials"]}회 분기·{result["audit"]["radio_frames"]:,}개 전파 프레임을 재검증했다. '
        f'안전 위반 {result["audit"]["safety_violations"]}건. 패킷 보존과 기존 파일 지문을 확인했다.','',
        '시뮬레이션 계산 시간은 작업 중인 데스크톱에서 측정했다. 여러 실험을 동시에 실행한 측정값에는 '
        'CPU 경합이 포함된다. 0.1초 감시와 전체 후보 경로 재계산은 별개이며, 전체 계획은 한 번에 한 작업만 수행하도록 0.2초를 점유한다.','',
        '| 구성 | 조건별 판단 p95의 중앙값(ms) | 계산 예산 초과 횟수 | 계산 중 요청 생략 |','|---|---:|---:|---:|']
    for a,s in stats.items():
        lines.append(f'| {LABELS[a]} | {s["median_trial_p95_decision_ms"]:.1f} | {s["decision_budget_exceeded"]}/{s["assessments"]} | {s["busy_skips"]} |')
    eligible=result['eligible_for_further_full_mission_study']
    lines+=['','## 채택 판단','',
        '별도 전체 임무 후속 검증 기준을 충족한 구성: '+(', '.join(LABELS[a] for a in eligible) if eligible else '없음')+'.',
        '이번 결과만으로 운영 Final V1을 교체하지 않았다. 빠른 제어가 도움이 되더라도 비컨·명령 비용, '
        '데이터 전달, 운동 제한과 계산 완료 시각을 함께 확인해야 한다. '
        '실제 장비의 비컨 크기·주파수·가속도·GPS 오차·추종 지연은 아직 정해지지 않았다.','',
        '![빠른 제어 비교](빠른제어_비교.png)','',
        '[수치 집계](artifacts/results.json) · [시나리오](artifacts/cases.json) · [구현 고정](artifacts/freeze.json) · [감사](artifacts/audit.json)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
