"""v3의 일반화 실패까지 포함하는 개발/최종 확인 분리 보고서."""
import base64,csv,json
from pathlib import Path
import numpy as np
from cooperative.deployment import STUDY,plan,verify_selection,VERSION
from cooperative.continuity_report import (LABELS as OLD_LABELS,ENGLISH as OLD_ENGLISH,
    extended_summary,independent_reward_audit,TERM_NAMES)
from cooperative.refinement_analysis import (trajectory_audit,city_for,safety_summary,
    NETWORK_COLUMNS,paired_spatial_interval)
from cooperative.io import validated_result
from cooperative.analyze import key,METRICS
from cooperative.refine import atomic_json
from cooperative.contracts import verify_forecaster_files
from swarm_sim.regions import ROOT,sha256_file

LABELS={**OLD_LABELS,'consensus_025':'정책 합의 · 새 정책 25%',
    'consensus_050':'정책 합의 · 새 정책 50%','consensus_075':'정책 합의 · 새 정책 75%'}
ENGLISH={**OLD_ENGLISH,'consensus_025':'Consensus (new 25%)','consensus_050':'Consensus (new 50%)','consensus_075':'Consensus (new 75%)'}
METRIC_NAMES=['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s',
              'duration_ratio','relay_travel_m','command_switch_count']


def audit_episode(prefix,spec,candidate,p,phase,tag):
    sha=sha256_file(ROOT/candidate['checkpoint']) if candidate['checkpoint'] else None
    row=validated_result(prefix,spec,sha)
    if (not row or row.get('deployment_sources')!=p['source_sha256'] or row.get('continuity_sources')!=p['base_source_sha256']
        or row['source_version']!=VERSION or row['controller']!=candidate['controller'] or row['teacher_profile']!=candidate['profile']):
        raise ValueError('Deployment raw/source/policy contract failed: '+str(prefix))
    risk=tag not in ['nominal','v2_no_risk_ppo']
    if row['risk_input_enabled']!=risk or (not risk and row['forecast_calls']!=0):raise ValueError('Confirmation risk input mismatch')
    with np.load(Path(prefix).with_suffix('.npz')) as z:
        city=city_for(row['region']);audit=trajectory_audit(z,city,row)
        audit.update(independent_reward_audit(z,city,row))
        row.update({k:float(z['C'][:,col].mean()) for k,col in NETWORK_COLUMNS.items()})
    if not audit['passed'] or not row['mission_completed']:raise ValueError('Deployment full mission safety failed')
    return row,dict(phase=phase,tag=tag,**spec,**audit)


def phase_audit(p,phase,candidates,specs):
    groups={};audits=[]
    for tag,candidate in candidates.items():
        rows=[];declared=json.loads((STUDY/phase/f'{tag}_complete.json').read_text(encoding='utf-8'))
        if declared['candidate']!=candidate or len(declared['results'])!=len(specs) or {key(r) for r in declared['results']}!={key(s) for s in specs}:
            raise ValueError('Deployment complete set mismatch')
        for spec in specs:
            stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
            row,audit=audit_episode(STUDY/phase/tag/stem,spec,candidate,p,phase,tag)
            original=next(r for r in declared['results'] if key(r)==key(spec))
            if original['array_sha256']!=row['array_sha256']:raise ValueError('Aggregate/raw mismatch')
            rows.append(row);audits.append(audit)
        groups[tag]=sorted(rows,key=key)
    return groups,audits


def analyze():
    p=plan();freeze=verify_selection(p);audits=[];development={}
    base_root=ROOT/'results/continuity'
    prior=json.loads((base_root/'audited_summary.json').read_text(encoding='utf-8'))
    if (prior['source_sha256']!=p['base_source_sha256'] or not prior['formal_safety']['all_passed']
        or prior['counts']['formal']!=636 or verify_forecaster_files()!=prior['forecaster_files_sha256']):
        raise ValueError('Audited v3 parent study changed')
    for name,sha in prior['completed_test_sha256'].items():
        if sha256_file(base_root/'test'/name)!=sha:raise ValueError('v3 parent test aggregates changed')
    for phase,specs in p['development'].items():
        groups,checked=phase_audit(p,'development_'+phase,p['blend_candidates'],specs);audits+=checked
        for tag,c in p['reference_candidates'].items():
            raw=json.loads((base_root/phase/f'{tag}_complete.json').read_text(encoding='utf-8'))
            if len(raw['results'])!=len(specs) or {key(r) for r in raw['results']}!={key(s) for s in specs}:raise ValueError('Parent development conditions changed')
            sha=sha256_file(ROOT/c['checkpoint']) if c['checkpoint'] else None
            rows=[]
            for s in specs:
                stem=f'{s["region"]}_{s["direction"]}_{s["kind"]}_{s["seed"]}'
                row=validated_result(base_root/phase/tag/stem,s,sha)
                if not row or row['continuity_sources']!=p['base_source_sha256']:raise ValueError('Parent raw data changed')
                rows.append(row)
            groups[tag]=sorted(rows,key=key)
        development[phase]={tag:extended_summary(rows) for tag,rows in groups.items()}
    scores={};feasible={}
    for tag in freeze['candidates']:
        ds=[development[phase][tag]['macro']['search_connected_mean']-development[phase]['v2_ppo']['macro']['search_connected_mean'] for phase in development]
        scores[tag]=[min(ds),float(np.mean(ds)),float(np.mean([development[phase][tag]['macro']['timely_pdr'] for phase in development]))]
        feasible[tag]=all(development[phase][tag]['complete_count']==len(p['development'][phase]) and
            development[phase][tag]['safety_violations']==0 and development[phase][tag]['max_duration_ratio']<=1.15 and
            development[phase][tag]['macro']['pdr']>=development[phase]['v2_ppo']['macro']['pdr']-.001 for phase in development)
    chosen=max([k for k in scores if feasible[k]],key=lambda k:tuple(scores[k]))
    if chosen!=freeze['selected_candidate'] or feasible!=freeze['feasible'] or any(not np.allclose(scores[k],freeze['robust_score'][k],rtol=0,atol=1e-12) for k in scores):
        raise ValueError('Robust development selection cannot be reproduced')
    groups,confirmation_audits=phase_audit(p,'confirmation',freeze['confirmation_candidates'],p['confirmation']);audits+=confirmation_audits
    summaries={k:extended_summary(v) for k,v in groups.items()}
    selected=summaries[chosen];reference=summaries['v2_ppo']
    passed=bool(selected['complete_count']==32 and selected['safety_violations']==0 and selected['max_duration_ratio']<=1.15
        and selected['macro']['search_connected_mean']>=reference['macro']['search_connected_mean']
        and selected['macro']['pdr']>=reference['macro']['pdr']-.0005)
    active=chosen if passed else 'v2_ppo'
    maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    spatial={m['key']:m['spatial_group'] for m in maps}
    def comparisons(base):return {tag:{m:paired_spatial_interval(rows,groups[base],m,spatial) for m in METRIC_NAMES} for tag,rows in groups.items() if tag!=base}
    all_audits=json.loads((base_root/'all_formal_trajectory_audit.json').read_text(encoding='utf-8'))+audits
    if len(audits)!=372 or len(all_audits)!=1008:raise ValueError('Required 636 prior +180 blend development +192 confirmation missions missing')
    records=[]
    for tag,rows in groups.items():
        for r in rows:
            record=dict(method=tag,spatial_group=spatial[r['region']],**r)
            record.update({'cost_'+k:r['reward_terms_sum'][k] for k in TERM_NAMES});records.append(record)
    fields=['method','spatial_group','region','direction','kind','seed','mission_completed',*METRICS,*NETWORK_COLUMNS,
            'reward_sum','command_switch_count','relay_stationary_fraction',*['cost_'+k for k in TERM_NAMES],
            'generated_packets','delivered_packets','retries','ttl_drops','retry_drops','overflow_drops','beacon_tx','beacon_rx',
            'forecast_calls','risk_input_enabled','controller','teacher_profile','checkpoint_sha256','array_sha256']
    with (STUDY/'confirmation_all_metrics_v3_1.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(fields)),extrasaction='ignore');w.writeheader();w.writerows(records)
    result=dict(format='audited-deployment-confirmation-v3.1',selected_candidate=chosen,active_default=active,
        selected_policy=freeze['candidates'][active],promotion_gate_passed=passed,new_policy_promoted=passed and chosen!='v2_ppo',
        selection_sha256=sha256_file(STUDY/'selection.json'),source_sha256=p['source_sha256'],
        parent_v3_summary_sha256=sha256_file(base_root/'audited_summary.json'),
        forecaster_files_sha256=verify_forecaster_files(),development_summaries=development,robust_scores=scores,
        confirmation_summaries=summaries,vs_nominal=comparisons('nominal'),vs_v2_ppo=comparisons('v2_ppo'),
        counts=dict(training=96,development=720,development_existing_v3=540,development_new_blends=180,
                    confirmation=192,formal_total=1008,pilot=48,final_demos=6,historical_guard_demos=6),
        safety=safety_summary(confirmation_audits),formal_safety=safety_summary(all_audits),
        reward_reconstruction=dict(all_passed=True,max_error=max(a['max_reward_term_error'] for a in all_audits)),
        completed_confirmation_sha256={f.name:sha256_file(f) for f in sorted((STUDY/'confirmation').glob('*_complete.json'))},
        previous_v3_test_used_as_development=True,confirmation_used_for_candidate_selection=False,
        confirmation_used_for_predeclared_promotion_gate=True,regions=8,spatial_groups=6,
        limitations=['existing eight geographic test maps reused; fresh confirmation seeds, not first geographic external test',
            'spatial group intervals condition on fitted models; no retraining seed uncertainty',
            'no separately trained no-risk v3 PPO; no risk-only causal gain claim',
            'fixed LightGBM and ideal routing/shared safety/zero simulated compute latency; hardware and full 3GPP not validated'])
    atomic_json(STUDY/'new_trajectory_audit.json',audits)
    atomic_json(STUDY/'all_formal_trajectory_audit.json',all_audits)
    atomic_json(STUDY/'audited_summary.json',result)
    print(json.dumps(dict(candidate=chosen,active=active,promotion_gate_passed=passed,formal=result['formal_safety']),ensure_ascii=False),flush=True)
    return result


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'docs/assets/deployment';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    tags=[t for t in data['confirmation_summaries'] if t!='nominal']
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for i,k in enumerate(tags):
        r=data['vs_nominal'][k]['search_connected_mean'];c='#0b7b81' if k==data['active_default'] else '#8795a6'
        ax.hlines(i,100*r['ci95_low'],100*r['ci95_high'],color=c,lw=2.4);ax.scatter(100*r['difference'],i,color=c,s=46,zorder=3)
    ax.set_yticks(range(len(tags)),[ENGLISH[k] for k in tags]);ax.invert_yaxis();ax.axvline(0,color='#af5d47',ls='--',lw=1)
    ax.grid(axis='x',alpha=.18);ax.set_xlabel('Connectivity change vs nominal (percentage points)')
    ax.set_title('Untouched confirmation · 32 scenarios per policy · 6 spatial groups',loc='left',pad=14)
    a=folder/'confirmation_intervals.png';fig.savefig(a,dpi=170);fig.savefig(a.with_suffix('.svg'));plt.close(fig)
    tags=[k for k in data['vs_v2_ppo'] if k!='nominal']
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for i,k in enumerate(tags):
        r=data['vs_v2_ppo'][k]['search_connected_mean'];c='#0b7b81' if k==data['selected_candidate'] else '#8795a6'
        ax.hlines(i,100*r['ci95_low'],100*r['ci95_high'],color=c,lw=2.4);ax.scatter(100*r['difference'],i,color=c,s=46,zorder=3)
    ax.set_yticks(range(len(tags)),[ENGLISH[k] for k in tags]);ax.invert_yaxis();ax.axvline(0,color='#af5d47',ls='--',lw=1)
    ax.grid(axis='x',alpha=.18);ax.set_xlabel('Connectivity change vs previous PPO (percentage points)')
    ax.set_title('Extra gain beyond v2 · paired spatial-group intervals',loc='left',pad=14)
    b=folder/'previous_ppo_intervals.png';fig.savefig(b,dpi=170);fig.savefig(b.with_suffix('.svg'));plt.close(fig)
    return a,b


def _uri(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
def interval(r):return f'{r["difference"]*100:+.3f}%p [{r["ci95_low"]*100:+.3f}, {r["ci95_high"]*100:+.3f}]'


def build():
    from cooperative.report import font_css
    d=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'));f=verify_selection(plan())
    active=d['active_default'];candidate=d['selected_candidate'];main=d['confirmation_summaries'][active]['macro'];base=d['confirmation_summaries']['nominal']['macro']
    a,b=figures(d)
    rows=[]
    for tag,s in d['confirmation_summaries'].items():
        m=s['macro'];rows.append('<tr'+(' class="chosen"' if tag==active else '')+'>'+f'<th>{LABELS[tag]}'+(' · 기본값' if tag==active else '')+'</th>'+
            f'<td>{m["search_connected_mean"]*100:.3f}%</td><td>{m["pdr"]*100:.3f}%</td><td>{m["outage_total_s"]:.1f}</td><td>{m["outage_max_s"]:.1f}</td>'+
            f'<td>{m["duration_ratio"]:.3f}</td><td>{m["relay_travel_m"]/1000:.2f}</td><td>{m["command_switch_count"]:.0f}</td><td>{s["complete_count"]}/32</td></tr>')
    dev_rows=[]
    for tag in f['candidates']:
        va=d['development_summaries']['validation'][tag]['macro'];te=d['development_summaries']['test'][tag]['macro']
        dev_rows.append(f'<tr><th>{LABELS[tag]}</th><td>{va["search_connected_mean"]*100:.3f}%</td><td>{te["search_connected_mean"]*100:.3f}%</td><td>{d["robust_scores"][tag][0]*100:+.3f}%p</td></tr>')
    extra=d['vs_v2_ppo'].get(candidate,{}).get('search_connected_mean')
    gain=interval(extra) if extra else '이전 PPO와 동일'
    evidence=('개선 방향을 관측했다.' if extra and extra['ci95_low']>0 else
              '구간이 0을 포함하므로 이전 PPO를 확실히 개선했다고 단정할 수 없다.' if extra and extra['ci95_high']>=0 else
              '이전 PPO보다 낮은 방향을 관측했다.' if extra else '기존 정책을 유지했다.')
    decision='사전 승격 기준을 통과해 후보를 기본값으로 적용했다.' if d['new_policy_promoted'] else '사전 승격 기준에 따라 이전 PPO를 기본값으로 유지했다.'
    best=max(d['confirmation_summaries'],key=lambda k:d['confirmation_summaries'][k]['macro']['search_connected_mean'])
    css='''*{box-sizing:border-box}body{margin:0;background:#edf3f5;color:#18323e;font:16px/1.8 ReportKR,"Malgun Gothic",Arial,sans-serif}main{max-width:1200px;margin:auto;padding:34px 24px 60px}header{background:#153946;color:#fff;padding:34px;border-radius:20px}h1{font-size:33px;line-height:1.4;margin:10px 0}h2{font-size:23px;margin-top:0}.muted{color:#52717e}header p{color:#cde4ea}.tag{font-size:12px;letter-spacing:.13em;color:#9edfe1}section{margin-top:22px;padding:28px;background:white;border:1px solid #d8e4e8;border-radius:16px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:22px}.card{background:#e6f4f3;padding:20px;border-radius:12px}.card b{font-size:29px;display:block;color:#0b7479}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dfe9eb;padding:11px;text-align:right;white-space:nowrap}th:first-child{text-align:left}thead{background:#edf5f6}.chosen{background:#e2f5ef}.note{padding:17px;background:#fff4e4;border-left:4px solid #dca044;border-radius:5px}img{display:block;width:100%;height:auto;margin:18px 0}code{background:#edf2f5;padding:2px 5px;border-radius:4px;font-size:13px;overflow-wrap:anywhere}pre{background:#16343f;color:#e8f6fa;padding:20px;border-radius:10px;overflow:auto}pre code{background:none;color:inherit}summary{cursor:pointer;font-weight:700}details{margin-top:14px}footer{padding-top:25px;font-size:12px;color:#627985;overflow-wrap:anywhere}@media(max-width:800px){main{padding:14px}.cards{grid-template-columns:1fr}header,section{padding:20px}h1{font-size:26px}}'''
    body=f'''<header><div class="tag">CAPSTONE DESIGN 2 · DEPLOYMENT CONFIRMATION · V3.1</div><h1>통신 우선 학습·재배치<br>개발과 최종 확인을 분리한 결과</h1><p>교수님 피드백을 반영하고 추가 PPO 학습·정책 결합을 비교한 뒤 새로운 난수로 최종 확인했다. 현재 기본값은 <strong>{LABELS[active]}</strong>이다.</p></header>
<div class="cards"><div class="card">최종 확인 연결률<b>{main['search_connected_mean']*100:.3f}%</b>기존 비행 {base['search_connected_mean']*100:.3f}%</div><div class="card">실제 패킷 전달률<b>{main['pdr']*100:.3f}%</b>기존 비행 {base['pdr']*100:.3f}%</div><div class="card">공식 원시 임무 검수<b>1,008 / 1,008</b>학습96 · 개발720 · 확인192</div></div>
<section><h2>1. 현재 결론과 기본값</h2><p>개발 자료에서 고른 후보는 <strong>{LABELS[candidate]}</strong>이며 최종 확인에서 이전 PPO 대비 연결률 변화는 <strong>{gain}</strong>다. {evidence} {decision}</p><p>승격 기준은 확인 평균 연결률이 이전 PPO 이상, PDR이 이전 PPO−0.05%p 이상,32조건 모두 완주·실제 안전 위반0·최대 임무 시간1.15배 이내다. 이 기준은 확인 실행 전에 고정했다. 확인 결과를 보고 다른 후보를 다시 고르지 않았다.</p><p>최종 확인 평균의 수치상 최고는 {LABELS[best]} ({d['confirmation_summaries'][best]['macro']['search_connected_mean']*100:.3f}%)다. 최고 시험값과 사전 선정·승격 규칙을 구분한다.</p><p class="note">초기 v3의 가드 방식은 검증에서는1위였지만 이후 시험에서 기존 PPO보다 낮았다. 그 기록을 보존하고, 초기 검증/시험을 새 후보 개발 자료로 재분류했다. 아래 최종 성능은 이 개발에 사용하지 않은 새32조건의 확인 결과만 집계했다. 지리 지도8곳은 이미 개발 중 확인한 기존 지도다. 새 조건 난수의 확인이며 지리적 독립 외부 검증은 아니다.</p></section>
<section><h2>2. 새 난수에서의 최종 확인 · 6방식 ×32조건</h2><p>지도별4조건 평균을8지도에 동일 가중치로 평균했다. 수색3대가 중계 경유를 포함해 모두 상호 도달 가능한0.5초 구간의 비율을 연결 유지율로 정의한다. 단절 항은 임무별 총 단절/최장 단절을 지역 평균한 값이다.</p><div class="table"><table><thead><tr><th>방식</th><th>연결률</th><th>PDR</th><th>총 단절(s)</th><th>최장 단절(s)</th><th>시간 배율</th><th>중계 이동(km)</th><th>명령 전환</th><th>완주</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><img alt="최종 확인 연결률 변화와 공간 그룹 구간" src="{_uri(a)}"><img alt="이전 PPO 대비 추가 효과와 공간 그룹 구간" src="{_uri(b)}"><p>구간은 인접·중첩 지도를6개 공간 그룹으로 묶은 대응 부트스트랩10,000회다. 적은 그룹에서 얻는 설명적 구간이며 학습 난수를 다시 바꾸는 불확실성이나 다중 비교 전체의 오류율까지 반영하지 않는다. 최종 후보와 그 승격 기준은 확인 전에 고정했다.</p></section>
<section><h2>3. 교수님 피드백과 MADDPG에서 반영한 내용</h2><ul><li>누적 보상은 설계 스케일에 따라 달라지므로 기본값을 보상합으로 선정하지 않는다. 연결률·PDR·단절 초·완주·실제 이동량을 별도로 비교한다.</li><li>주요 보상 비용은 단절1초=1. 3초당 추가 이동 비용은 최대0.003. 0.5초 단절 한 번의 비용은 최대 이동 비용의166.7배다.</li><li>긴 단절 추가 비용과 해당3초 생성 패킷의 최종 전달/지연 품질을 반영한다. 양의 생존 보상이 없어 임무 연장으로 보상을 모을 수 없다.</li><li>첨부의 폐루프·지연 패킷 정산·중앙 학습/분산 실행·명령 유지 아이디어를 검토했다. 실제 미래 lookahead·이상적 전역 Actor 관측·큰 사전 H 벌점은 도입하지 않았다.</li><li>첨부의 관측112/연속2D 행동 가중치를 현재 문맥91/후보25개 정책에 직접 연결하지 않았다. 현재 Actor3개를 초기값으로 새 전체 임무32개씩 실제 추가 학습했다.</li><li>건물·금지 구역·연속5m 분리·속도5m/s 제한은 기존 공통 이동기가 적용한다. 보상으로 안전을 증명했다는 의미가 아니다.</li></ul><p>첨부 CSV20행 재계산에서 MADDPG L2 0.05는 통신이 개선돼도 H 항−141.75 때문에 누적 보상이 낮았다. 그 첨부와 본 실험은 지도/조건/입력 계약이 달라 평균을 직접 비교할 수 없다. 상세 반영표·원본 텍스트·재계산 스크립트는 <code>docs/MADDPG_교수피드백_반영검토.md</code>와 <code>docs/maddpg_review/</code>에 있다.</p></section>
<section><h2>4. 추가 학습과 검토한 정책 결합</h2><p>고정 LightGBM 예측기 → 수신 비콘/ACK 이력과 정적 지도 → 로컬 후보25개 → 공유 PPO → 공통 안전 이동의 폐루프다. 학습용 중앙 Critic만 전역 상태를 쓴다. v2의 세 위험 사용 Actor만 초기값으로 재사용하고 Critic/Adam은 새로 초기화했다. 난수28701/28702/28703마다32개 새 전체 임무,4회 갱신을 수행하고 세 정책 logit을 평균한다.</p><p>정책 합의는 기존/새 정책 각각의 관측·교사점수 계약을 유지한 채, 두 정책의 후보 확률을 혼합한다. 새 정책 비율25/50/75%를 개발 자료에서 비교했다. 두 정책이 모두 동의한 행동은 보존되며, 행동마다 실제 미래 채널을 조회하지 않는다. 합의 정책에는 추가 역전파를 하지 않았고, 고정된 두 학습 정책의 실행 결합이다.</p><p>개발 선정은 두 개발 단계 각각에서 완주·안전·시간 제한과 이전 PPO 대비 PDR−0.1%p 이내를 만족하는 후보에 대해, 이전 PPO 대비 연결률 개선의 두 단계 중 작은 값을 최대화했다. 한 개발 단계에서만 좋은 가드를 피하기 위한 규칙이다.</p><details><summary>개발 비교 전체 · 최종 확인 성능과 구분</summary><div class="table"><table><thead><tr><th>방식</th><th>개발A 연결률</th><th>개발B 연결률</th><th>최악 단계의 이전 PPO 대비 차이</th></tr></thead><tbody>{''.join(dev_rows)}</tbody></table></div></details></section>
<section><h2>5. 검수와 결과의 적용 범위</h2><p>학습96 + 개발720 + 새 확인192 =1,008개 공식 임무를 검수했다. 학습용 후보48회와 새 기본값 시연6회, 초기 가드 시연6회는 별도다. 연속 이동 최소 분리 {d['formal_safety']['minimum_swept_separation_m']:.3f}m, 충돌·건물 침범·잘못된 간선 이동0건이다. 링크 행렬에서 전체10개 지표를 다시 계산하고, 실제 궤적/기준 경로/패킷 출생 코호트에서8항 보상을 재구성했다. 최대 항 오차는 {d['reward_reconstruction']['max_error']:.2e}다.</p><ul><li>N8(수색3/중계5), 명목V40m,400m×400m,고도50m,RF0.5초/제어3초. 수색 드론 간 전달이 목적이며 GCS 목적지·실제 조종 통신은 포함하지 않는다.</li><li>ITU FSPL·Bullington 회절 근사·시간 상관 섀도잉·간섭·BPSK·이상적TDMA다. 전체3GPP TR36.777 구현은 아니다.</li><li>이상적 전역 라우팅·공동 안전 이동·동기화·계산 지연 없는 시뮬레이션 시간 가정이 남는다. 기체·Wi-Fi 스택·전력·GPS·바람·지연 검증은 별도다.</li><li>예측기는 고정이며 정책과 공동 역전파하는 엄밀한E2E 학습은 아니다. 새 위험 제거 PPO를 별도로 재학습하지 않아 새 PPO 위험 입력만의 인과 효과를 주장하지 않는다.</li><li>추가32임무×3난수와 정책 혼합은 전역 최적 수렴·모든 환경에서 최고 성능을 증명하지 않는다. 후보군과 현재 모델 범위의 실측 비교다.</li><li>Python3.12/Linux CPU 검증. Windows용 BAT 제공, 실제 Windows 실행 검증은 별도다. HTML은 브라우저 파일 정책으로 상호작용 검수를 수행하지 못했고 정적 HTML/JS/프레임 데이터로 확인했다.</li></ul></section>
<section><h2>6. 실행과 이어서 연구</h2><pre><code>python cooperate_run.py doctor
python cooperate_run.py verify
python cooperate_run.py compare --region gangnam --seed 2026 --out results/deployment/custom/gangnam
python cooperate_run.py deployment --stage all --workers 4
python cooperate_run.py finalize --workers 4</code></pre><p>현재 기본값은 <code>results/deployment/published_selection.json</code>이 기준이다. 후보 동결은 <code>results/deployment/selection.json</code>, 확인 원본과 CSV는 <code>results/deployment/confirmation/</code>과 <code>confirmation_all_metrics_v3_1.csv</code>다. 초기 v3와 기존 v2의 핵심 소스·모델·실험 기록을 별도로 보존했다. 다음 대화에서는 ZIP과 <code>CONTINUE_HERE.md</code>를 먼저 읽는다.</p><p>원 논문: <a href="https://arxiv.org/abs/1707.06347">PPO</a>, <a href="https://arxiv.org/abs/1706.02275">MADDPG</a>. 현재 선택을 알고리즘 이름만으로 정당화하지 않는다.</p></section><footer>확인 전 후보 동결 SHA256: {d['selection_sha256']}<br>이전 v3 시험은 새 연구의 개발 자료로 사용했으며, 확인 난수는 configs/deployment_plan.json에 고정했다.</footer>'''
    if not candidate.startswith('consensus_'):
        body=body.replace('합의 결합에는 추가 역전파를 하지 않았고, 고정된 두 학습 정책의 실행 결합이다.', '합의 결합에는 추가 역전파를 하지 않았다. 세 혼합 비중은 일관된 개발 이득을 보이지 않아 최종 후보로 채택하지 않았다.')
    text='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>통신 우선 학습·재배치 최종 확인 v3.1</title><style>'+font_css(body+css)+css+'</style></head><body><main>'+body+'</main></body></html>'
    path=ROOT/'docs/통신우선_최종확인_v3_1.html';path.write_text(text,encoding='utf-8');return path


if __name__=='__main__':analyze();print(build())
