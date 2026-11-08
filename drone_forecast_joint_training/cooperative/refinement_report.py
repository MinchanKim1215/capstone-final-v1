"""확정된 원시 결과만 사용하는 독립 HTML·과학 그림 보고서."""

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cooperative.refine import STUDY
from cooperative.report import font_css
from swarm_sim.regions import ROOT

LABELS = {
    'nominal':'기존 비행', 'reactive':'현재 상태 반응형', 'mpc':'예측 위험 MPC',
    'mpc_no_risk':'위험값 없는 MPC', 'legacy_ppo':'이전 PPO',
    'risk_ensemble':'새 PPO · 예측 위험', 'risk_ensemble_guarded':'새 PPO · 예측 위험 + 가드',
    'no_risk_ensemble':'새 PPO · 위험값 제거', 'no_risk_ensemble_guarded':'새 PPO · 위험값 제거 + 가드',
}
ENGLISH = {
    'nominal':'Nominal flight','reactive':'Reactive','mpc':'Predictive MPC',
    'mpc_no_risk':'MPC without learned risk','legacy_ppo':'Previous PPO',
    'risk_ensemble':'PPO ensemble + risk','risk_ensemble_guarded':'Guarded PPO + risk',
    'no_risk_ensemble':'PPO ensemble, no risk','no_risk_ensemble_guarded':'Guarded PPO, no risk',
}
ORDER=['nominal','reactive','mpc_no_risk','mpc','legacy_ppo','no_risk_ensemble','risk_ensemble','no_risk_ensemble_guarded','risk_ensemble_guarded']


def png_uri(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def figures(data):
    folder=ROOT/'docs/assets/refinement';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=[x for x in ORDER if x!='nominal']
    fig,ax=plt.subplots(figsize=(9.2,5.1),layout='constrained')
    for i,tag in enumerate(methods):
        row=data['vs_nominal'][tag]['search_connected_mean']
        color='#096f82' if 'risk_ensemble' in tag and not tag.startswith('no_') else '#788c9f'
        ax.hlines(i,row['ci95_low']*100,row['ci95_high']*100,color=color,lw=2.2)
        ax.scatter(row['difference']*100,i,color=color,s=42,zorder=3)
    ax.set_yticks(range(len(methods)),[ENGLISH[x] for x in methods]);ax.invert_yaxis()
    ax.axvline(0,color='#9e493c',ls='--',lw=1)
    ax.set_xlabel('Search connectivity change vs nominal (percentage points)')
    ax.set_title('Paired test comparison · 32 scenarios · 6 spatial groups',loc='left',pad=14)
    ax.grid(axis='x',alpha=.17)
    p=folder/'connectivity_intervals.png';fig.savefig(p,dpi=160);plt.close(fig)
    paths={'intervals':p}
    fig,ax=plt.subplots(figsize=(8.2,3.0),layout='constrained')
    ablations=[('risk_vs_no_risk','PPO ensemble'),('risk_vs_no_risk_guarded','Guarded PPO ensemble'),('mpc_risk_vs_no_risk','MPC')]
    for i,(tag,label) in enumerate(ablations):
        row=data['matched_forecast_ablation'][tag]['search_connected_mean']
        ax.hlines(i,row['ci95_low']*100,row['ci95_high']*100,color='#096f82',lw=2.5)
        ax.scatter(row['difference']*100,i,color='#096f82',s=48,zorder=3)
    ax.set_yticks(range(3),[label for _,label in ablations]);ax.invert_yaxis()
    ax.axvline(0,color='#9e493c',ls='--',lw=1);ax.grid(axis='x',alpha=.17)
    ax.set_xlabel('Learned-risk input contribution (percentage points)')
    ax.set_title('Matched architecture · separately trained PPO controls',loc='left',pad=12)
    p=folder/'forecast_ablation.png';fig.savefig(p,dpi=160);plt.close(fig);paths['ablation']=p
    return paths


def number(value,digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def build():
    data=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    selection=json.loads((STUDY/'selection.json').read_text(encoding='utf-8'))
    paths=figures(data)
    selected=selection['selected'];main=data['summaries'][selected];base=data['summaries']['nominal']
    metric=main['macro'];b=base['macro'];rows=[]
    for tag in ORDER:
        item=data['summaries'][tag];m=item['macro']
        rows.append('<tr'+(' class="chosen"' if tag==selected else '')+'>'+
                    f'<th>{html.escape(LABELS[tag])}'+(' <small>검증 선정</small>' if tag==selected else '')+'</th>'+
                    f'<td>{number(m["search_connected_mean"]*100)}%</td><td>{number(m["pdr"]*100)}%</td>'+
                    f'<td>{number(m["outage_total_s"],1)}</td><td>{number(m["outage_max_s"],1)}</td>'+
                    f'<td>{number(m["duration_ratio"],3)}</td><td>{number(m["relay_travel_m"]/1000,2)}</td>'+
                    f'<td>{item["complete_count"]}/{item["episodes"]}</td></tr>')
    ablation_rows=[]
    for key,label in [('risk_vs_no_risk','PPO'),('risk_vs_no_risk_guarded','가드 적용 PPO'),('mpc_risk_vs_no_risk','MPC')]:
        r=data['matched_forecast_ablation'][key]['search_connected_mean']
        result='평균 개선 관측' if r['ci95_low']>0 else '평균 저하 관측' if r['ci95_high']<0 else '방향 확정 어려움'
        ablation_rows.append(f'<tr><th>{label}</th><td>{r["difference"]*100:+.3f}%p</td><td>[{r["ci95_low"]*100:+.3f}, {r["ci95_high"]*100:+.3f}]%p</td><td>{result}</td></tr>')
    seed_rows=[]
    for r in data['training_audit']:
        diag=selection['summaries'][f'{r["mode"]}_s{r["seed"]}']['macro']['search_connected_mean']
        seed_rows.append(f'<tr><th>{"예측 위험" if r["mode"]=="risk" else "위험값 제거"}</th><td>{r["seed"]}</td><td>{r["episodes"]}</td><td>{diag*100:.2f}%</td><td>{r["max_initial_logp_error"]:.2e}</td><td>{r["actor_change_l2"]:.3f}</td></tr>')
    safety=data['safety']
    val_count=sum(v['episodes'] for v in selection['summaries'].values())
    train_count=sum(r['episodes'] for r in data['training_audit'])
    total=train_count+val_count+data['test_episodes']
    names={m['key']:m for m in json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']}
    regional=[]
    for region,m in main['by_region'].items():
        old=base['by_region'][region];entry=names[region]
        regional.append(f'<tr><th>{html.escape(entry["name"])}</th><td>{entry["spatial_group"]}</td><td>{old["search_connected_mean"]*100:.2f}%</td><td>{m["search_connected_mean"]*100:.2f}%</td><td>{(m["search_connected_mean"]-old["search_connected_mean"])*100:+.3f}%p</td><td>{(m["pdr"]-old["pdr"])*100:+.3f}%p</td></tr>')
    region_table='<details><summary>지역별 선정 방식의 변화 확인</summary><div class="table"><table class="regional"><thead><tr><th>지역</th><th>공간 그룹</th><th>기존 연결률</th><th>선정 연결률</th><th>연결률 변화</th><th>전달률 변화</th></tr></thead><tbody>'+''.join(regional)+'</tbody></table></div></details>'
    network_rows=[]
    for label,key,scale,unit in [('전체 8대 연결률','all_drones_connected_mean',100.,'%'),
                                 ('평균 컴포넌트 수','mean_components',1.,'개'),
                                 ('최대 컴포넌트 평균 크기','largest_component_mean',8.,'대'),
                                 ('평균 노드 차수','mean_degree',1.,''),
                                 ('고립 드론 비율','isolated_drone_fraction',100.,'%'),
                                 ('네트워크 효율','network_efficiency_mean',100.,'%'),
                                 ('도달 가능한 수색 쌍의 최단 홉','search_shortest_hops_mean',1.,'홉')]:
        old,value=b.get(key),metric.get(key)
        network_rows.append(f'<tr><th>{label}</th><td>{number(old*scale if old is not None else None)}{unit}</td><td>{number(value*scale if value is not None else None)}{unit}</td></tr>')
    network_table='<details><summary>전체 8대 통신망의 추가 지표</summary><div class="table"><table><thead><tr><th>지표</th><th>기존 비행</th><th>선정 방식</th></tr></thead><tbody>'+''.join(network_rows)+'</tbody></table></div><p class="fine">전체 8대 연결률과 수색 3대의 중계 경로 연결률은 다른 지표. 링크 행렬에서 전체 10개 그래프 지표를 독립 재계산해 원본과 대조.</p></details>'
    best_test=max(data['summaries'],key=lambda tag:data['summaries'][tag]['macro']['search_connected_mean'])
    interpretation=f'이번 시험에서 연결 유지율의 수치상 최고는 {LABELS[best_test]} ({data["summaries"][best_test]["macro"]["search_connected_mean"]*100:.2f}%). 실행 기본값은 시험 이전 검증에서 선정한 {LABELS[selected]}로 유지.'
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>군집 UAV 예측·학습·재배치 통합 검증</title><style>
__FONT__
:root{--ink:#182d3b;--muted:#566a76;--line:#dce5e9;--accent:#096f82;--paper:#f3f6f7}*{box-sizing:border-box}body{margin:0;background:var(--paper);font-family:ReportKR,system-ui,"Malgun Gothic",sans-serif;color:var(--ink);line-height:1.72}main{max-width:1180px;margin:0 auto;padding:36px 26px 70px}.eyebrow{font-size:12px;letter-spacing:.16em;color:var(--accent);font-weight:700}h1{font-size:clamp(26px,4vw,40px);line-height:1.3;margin:10px 0 18px;letter-spacing:-.04em}h2{font-size:22px;margin:0 0 14px}h3{font-size:16px}p{margin:8px 0}header{padding:24px 0 26px}.lead{font-size:17px;color:var(--muted);max-width:900px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card,section{background:white;border:1px solid var(--line);border-radius:12px}.card{padding:18px}.card span{font-size:12px;color:var(--muted);display:block}.card strong{font-size:27px;display:block;margin:6px 0}.card small{font-size:12px;color:var(--muted)}section{padding:26px;margin:20px 0}.note{padding:14px 18px;border-left:3px solid var(--accent);background:#edf6f7;color:#274b58;margin:14px 0}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:11px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child{text-align:left}.regional th:first-child{white-space:normal;min-width:180px;max-width:360px}thead{background:#f2f5f6}tr.chosen{background:#edf7f8}th small{font-size:10px;display:block;color:var(--accent)}img{display:block;width:100%;height:auto;margin:15px auto;border:1px solid #eef2f4;border-radius:6px}ul,ol{padding-left:23px}.fine{font-size:12px;color:var(--muted)}a{color:#066b80;text-underline-offset:3px}code{background:#edf1f3;padding:3px 5px;border-radius:4px}pre{background:#f0f4f6;padding:16px;overflow:auto;font-size:12px;line-height:1.7;border-radius:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}summary{cursor:pointer;font-weight:650;padding:8px 0}@media(max-width:700px){main{padding:18px 14px 45px}section{padding:18px}.cards{grid-template-columns:1fr 1fr}.card strong{font-size:22px}.grid{grid-template-columns:1fr}table{font-size:12px}}
</style><main><header><div class="eyebrow">CAPSTONE · CLOSED-LOOP VALIDATION</div><h1>예측·학습·재배치 통합 검증</h1><p class="lead">실제 도시 지도와 같은 통신·임무 조건에서 비교한 결과. 정책 학습, 검증 선정, 새 난수 조건 시험을 구분.</p><div class="note"><b>검증 선정: __SELECTED__.</b> 선정은 검증 지역에서 동결. 아래 시험 수치로 정책을 다시 고르지 않는 절차.</div><div class="cards"><div class="card"><span>시험 연결 유지율</span><strong>__CTR__%</strong><small>기존 비행 대비 __DELTA_C__%p</small></div><div class="card"><span>시험 패킷 전달률</span><strong>__PDR__%</strong><small>기존 비행 대비 __DELTA_P__%p</small></div><div class="card"><span>선정 방식 임무 완주</span><strong>__COMPLETE__/32</strong><small>전체 임무 종료까지 평가</small></div><div class="card"><span>공식 학습·평가 실행</span><strong>__TOTAL__회</strong><small>학습 __TRAIN__ · 검증/진단 __VAL__ · 시험 __TEST__</small></div></div></header>
<section><h2>1. 공통 조건 비교</h2><p>시험 지도 8곳 × 2방향 × 2간섭 조건, 방식별 32회. 연결 유지율은 수색 드론 3대가 중계망을 통해 연결된 시간 비율. 지역별 평균을 같은 비중으로 집계.</p><div class="table"><table><thead><tr><th>방식</th><th>연결 유지율 ↑</th><th>전달률 ↑</th><th>누적 단절 초 ↓</th><th>최장 단절 초 ↓</th><th>임무 시간 배율 ↓</th><th>중계 이동 km ↓</th><th>완주</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="fine">위험값 없는 방식도 지도와 관측된 속도의 기하학적 외삽을 사용. 제거한 정보는 학습된 통신 위험·단절 시작 확률. 이전 PPO는 과거 학습 경로의 비교 기준이며 새 배포 후보와 별도.</p><img src="__INTERVAL_IMAGE__" alt="기존 비행 대비 각 방식의 연결 유지율 변화와 공간 그룹 대응 구간"><p class="fine">점: 기존 비행 대비 평균 변화. 선: 독립 공간 그룹 6개를 단위로 대응 재표집한 서술적 95% 구간. 서로 겹치는 지도를 독립 표본으로 중복 계산하지 않는 방식. 표본 그룹이 적어 정밀한 확정 결론에는 한계.</p></section>
<section><h2>2. 예측 위험의 기여</h2><p>위험 입력이 있는 PPO와 없는 PPO를 각각 처음부터 학습. 같은 학습 난수·초기 가중치·임무 일정·안전 조건을 사용하고, Actor와 Critic 모두에서 학습된 위험 입력을 제거한 대조군. 단일 정책의 입력만 사후에 지우는 방식과 구분.</p><div class="table"><table><thead><tr><th>같은 제어 구조</th><th>위험 입력에 따른 변화</th><th>공간 그룹 95% 구간</th><th>해석</th></tr></thead><tbody>__ABLATION_ROWS__</tbody></table></div><img src="__ABLATION_IMAGE__" alt="같은 제어 구조에서 학습된 위험 입력의 기여"><p>가드는 PPO가 선택한 행동의 로컬 모델 점수가 해당 순간의 최선 후보보다 0.35를 초과해 낮으면 모델 기반 후보로 복귀하는 규칙. 실제 채널 정답·미래 간섭 시각은 사용하지 않는 구조. 가드 유무를 맞춰 예측 기여를 비교.</p></section>
<section><h2>3. 학습·선정·시험 절차</h2><div class="grid"><div><h3>모델과 입력</h3><ul><li>고정 LightGBM 예측기 → 실제 수신 관측의 위험값 → 로컬 이동 후보 → 공유 PPO 정책 → 안전 이동.</li><li>예측 유무별 독립 학습 난수 3개, 정책당 전체 임무 64회. 세 정책 logit을 평균하는 동일 가중치 앙상블.</li><li>중앙 Critic은 학습 전용. 실행 Actor는 로컬 수신 정보·자기 상태·정적 지도·자기 목표 사용.</li><li>예측기까지 함께 역전파하는 엄밀한 공동 E2E 학습과 구분. 이번 예측 가중치는 고정.</li></ul></div><div><h3>평가와 선정</h3><ul><li>지도 45곳: 학습 30 / 검증 7 / 기존 시험 8.</li><li>선정 후보는 검증 28조건에서 완주·안전·전달률·시간 제한을 통과한 방식.</li><li>전달률 저하 0.5%p 이내, 임무 시간 배율 최대 1.15 이내에서 연결 유지율 우선. 프로젝트 설계 기준이며 국제 표준의 합격 기준이 아님.</li><li>새 난수 시험 32조건은 검증 이전에 규칙으로 지정. 기존 시험 지도 재사용에 따른 새 조건 재평가이며, 새로 수집한 8개 도시의 최초 시험이라는 의미는 아님.</li></ul></div></div><details><summary>독립 학습 난수와 정책 갱신 검수</summary><p>PPO의 행동 수집 이후 외부에서 정책 잔차 크기를 바꾸는 코드를 제거. 모든 갱신 전에 수집 당시 log-probability와 현재 정책을 전체 표본에서 대조. 독립 정책의 14조건 진단 결과는 학습 난수 선택에 사용하지 않는 구성.</p><div class="table"><table><thead><tr><th>입력</th><th>학습 난수</th><th>전체 임무</th><th>검증 진단 연결률</th><th>최대 초기 logp 오차</th><th>행동망 변화 L2</th></tr></thead><tbody>__SEED_ROWS__</tbody></table></div></details></section>
<section><h2>4. 안전·무결성 검수</h2><p>시험 __TEST__회 전체의 저장 궤적에서 별도 재계산. 시뮬레이터가 기록한 안전 위반 수만 인용하지 않는 검사.</p><ul><li>드론 간 연속 이동 최소 거리: <b>__MINSEP__m</b> / 설정 5m.</li><li>연속 이동 충돌 쌍: <b>__COLLISIONS__</b>. 비행 제한 영역 교차 선분: <b>__OBSTACLES__</b>. 비행 그래프 외 이동: <b>__BAD_EDGES__</b>.</li><li>생성 패킷 = 전달 + TTL 폐기 + 재시도 폐기 + 큐 초과 폐기 보존식 확인.</li><li>원시 RF 기록에서 연결 유지율 재계산. 3초 이동, 0.5초 RF, 저장 프레임 수의 일관성 확인.</li><li>위험 입력 제거 학습에서 예측기 호출 0회, Actor·Critic 위험 특징 0 확인. 대응하는 학습의 초기 가중치 일치 확인.</li></ul><p class="fine">수치 상세: <a href="../results/refinement/comparison_all_metrics_v2.csv">실행별 전체 지표 CSV</a> · <a href="../results/refinement/audited_summary.json">검수 요약 JSON</a> · <a href="../results/refinement/trajectory_audit.json">시험 궤적 검수</a></p></section>
<section><h2>5. 구현 범위와 사용 방법</h2><p><b>전체 8대 · 수색 3대 · 중계 5대 · 400m×400m · 고도 50m · 명목 V 대형 40m.</b> 임의의 드론 수·고도·대형 전체에서의 최적 성능을 뜻하지 않는 고정 설계.</p><p>일부 건물 높이는 원본 데이터의 추정값을 포함. 채널은 기존 ITU 자유공간 손실·Bullington 지붕 회절 근사·시간 상관 섀도잉·외부 간섭·BPSK 성공 확률을 유지. 이상적 TDMA와 전역 라우팅, 공동 안전 이동기, 계산 지연을 추가하지 않는 시뮬레이션 시간 가정 포함. 3GPP TR 36.777 전체 구현이나 실제 Wi-Fi·실기체 운용 인증 결과와 구분.</p><div class="note">HTML 결과와 전체 임무 재생은 설치 없이 열람. 새 시뮬레이션·재학습은 Python 3.12와 ZIP 전체 해제 필요.</div><ol><li><code>00_환경설정.bat</code> — 검증한 CPU 의존성 설치와 환경 검사.</li><li><code>01_통합_결과.bat</code> — 결과와 세 지역 전체 임무 재생.</li><li><code>02_통합_시뮬레이션.bat</code> — 새 난수의 기존 비행/선정 방식 비교.</li><li><code>03_통합_검사.bat</code> — 자동 회귀 검사.</li></ol><pre>python cooperate_run.py doctor
python cooperate_run.py compare --region gangnam --seed 2026 --out results/refinement/custom/gangnam</pre><p><a href="통합_전체임무_재생_v2.html">세 지역 전체 임무 비교 재생</a> · <a href="../README.md">실행 안내</a></p></section>
<section><h2>6. 알고리즘 근거</h2><p>안전한 이산 후보 25개에서 행동을 고르는 공유 PPO와 중앙 학습용 가치함수. 팀원 MADDPG를 그대로 연결하는 대신 최신 관측·비콘·이동 계약을 유지하는 선택. PPO의 보편적 우위를 전제하지 않고 동일 환경의 반응형·MPC·이전 학습 정책과 실제 비교.</p><ul><li><a href="https://arxiv.org/abs/1707.06347">Schulman et al., Proximal Policy Optimization Algorithms</a> — 수집 정책과 갱신 정책의 확률 비율을 이용한 정책 최적화.</li><li><a href="https://arxiv.org/abs/2103.01955">Yu et al., The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games</a> — 협력 환경에서의 PPO와 구현 요소를 검토한 근거. 도시 UAV의 우위는 이번 실제 비교 결과로 판단.</li><li><a href="https://arxiv.org/abs/2304.01315">Patterson et al., Empirical Design in Reinforcement Learning</a> — 학습 변동성, 대응 비교, 환경·난수 구분과 평가 설계의 근거.</li></ul><p class="fine">정확한 보상 가중치·가드 한계·선정 제한은 프로젝트 설계. 문헌에서 보장한 최적값으로 표현하지 않는 범위.</p></section></main></html>'''
    replacements={
        '__SELECTED__':html.escape(LABELS[selected]),'__CTR__':number(metric['search_connected_mean']*100),
        '__PDR__':number(metric['pdr']*100),'__DELTA_C__':f'{(metric["search_connected_mean"]-b["search_connected_mean"])*100:+.3f}',
        '__DELTA_P__':f'{(metric["pdr"]-b["pdr"])*100:+.3f}','__COMPLETE__':str(main['complete_count']),
        '__TOTAL__':str(total),'__TRAIN__':str(train_count),'__VAL__':str(val_count),'__TEST__':str(data['test_episodes']),
        '__ROWS__':''.join(rows),'__ABLATION_ROWS__':''.join(ablation_rows),'__SEED_ROWS__':''.join(seed_rows),
        '__INTERVAL_IMAGE__':png_uri(paths['intervals']),'__ABLATION_IMAGE__':png_uri(paths['ablation']),
        '__MINSEP__':number(safety['minimum_swept_separation_m'],2),'__COLLISIONS__':str(safety['collision_pairs']),
        '__OBSTACLES__':str(safety['obstacle_segments']),'__BAD_EDGES__':str(safety['invalid_graph_edges']),
    }
    for key,value in replacements.items():text=text.replace(key,value)
    text=text.replace('<h2>1. 공통 조건 비교</h2>','<h2>1. 공통 조건 비교</h2><p class="note">'+html.escape(interpretation)+'</p>')
    formal=data['formal_safety']
    text=text.replace('<h2>4. 안전·무결성 검수</h2>',f'<h2>4. 안전·무결성 검수</h2><p class="note">학습·검증·시험 공식 {formal["episodes_checked"]:,}회 전체에도 동일한 독립 궤적·링크 검수 적용. 전체 구간 최소 드론 간격 {formal["minimum_swept_separation_m"]:.2f}m, 충돌 쌍 {formal["collision_pairs"]}개, 제한 영역 교차 {formal["obstacle_segments"]}개.</p>')
    text=text.replace('<img src="'+png_uri(paths['intervals'])+'"',region_table+network_table+'<img src="'+png_uri(paths['intervals'])+'"')
    csv_data=base64.b64encode((STUDY/'comparison_all_metrics_v2.csv').read_bytes()).decode()
    text=text.replace('href="../results/refinement/comparison_all_metrics_v2.csv"','download="comparison_all_metrics_v2.csv" href="data:text/csv;base64,'+csv_data+'"')
    text=text.replace('__FONT__',font_css(text))
    output=ROOT/'docs/통합_결과_v2.html';output.write_text(text,encoding='utf-8')
    return output


if __name__=='__main__':print(build())
