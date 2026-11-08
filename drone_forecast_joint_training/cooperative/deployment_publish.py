"""v3.1 사전 후보·확인 승격 기준·실행 기본값·HTML을 함께 게시."""
import argparse,json,shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import numpy as np
import torch
from cooperative.deployment import STUDY,plan,verify_selection,BLENDS
from cooperative.deployment_run import episode
from cooperative.deployment_report import LABELS,build as report,audit_episode
from cooperative.continuity_publish import archive_v2,data_index as v3_data_index
from cooperative.publish import check_frames
from cooperative.io import validated_result
from cooperative.refine import atomic_json
from swarm_sim.regions import ROOT,sha256_file

FORMAT='communications-deployment-published-v3.1'


def verify_publication():
    p=plan();f=verify_selection(p)
    pub=json.loads((STUDY/'published_selection.json').read_text(encoding='utf-8'))
    d=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    chosen=f['selected_candidate'];c=d['confirmation_summaries'][chosen];b=d['confirmation_summaries']['v2_ppo']
    gate=bool(c['complete_count']==32 and c['safety_violations']==0 and c['max_duration_ratio']<=1.15
        and c['macro']['search_connected_mean']>=b['macro']['search_connected_mean'] and c['macro']['pdr']>=b['macro']['pdr']-.0005)
    active=chosen if gate else 'v2_ppo'
    if (pub['format']!=FORMAT or pub['selected']!=active or d['active_default']!=active or pub['selected_candidate']!=chosen
        or pub['selected_policy']!=f['candidates'][active] or pub['source_sha256']!=p['source_sha256']
        or pub['selection_sha256']!=sha256_file(STUDY/'selection.json') or pub['audited_summary_sha256']!=sha256_file(STUDY/'audited_summary.json')
        or d['selection_sha256']!=pub['selection_sha256'] or d['formal_safety']['episodes_checked']!=1008
        or not d['formal_safety']['all_passed'] or not d['reward_reconstruction']['all_passed']
        or not pub['previous_v3_test_used_as_development'] or pub['confirmation_used_for_candidate_selection']):
        raise ValueError('Published deployment/default/confirmation provenance mismatch')
    return pub


def demo_plan(p):
    maps={m['key']:m for m in json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']}
    cases=[dict(region='teheran',seed=2026102801,kind='intermittent',direction='east_west')]
    for region,direction,kind in [('bucheon_jungdong_east','east_west','continuous'),('seoul_20261023_03','north_south','intermittent')]:
        cases.append(next(s.copy() for s in p['confirmation'] if (s['region'],s['direction'],s['kind'])==(region,direction,kind)))
    for s,name in zip(cases,['테헤란 · 학습 지역','부천 중동 · 최종 확인 지역','서울03 · 최종 확인 지역']):
        s.update(name=name,map=maps[s['region']]['path'],info='400m×400m · 전체 임무 · 사전 선정과 확인 승격 기준 적용 · v3.1',
            runs=[f'results/deployment/demos/{s["region"]}_{tag}' for tag in ['nominal','selected']])
    return dict(format='deployment-demos-v3.1',validation_selected=True,cases=cases,selection_by_confirmation_outcomes=False,
        case_rule='same three prior demo regions/direction/jammer strata; first prespecified new confirmation seed')


def capture_job(item):
    spec,c,prefix,sources=item;torch.set_num_threads(1)
    prefix=ROOT/prefix;ck=ROOT/c['checkpoint'] if c['checkpoint'] else None
    sha=sha256_file(ck) if ck else None
    old=validated_result(prefix,spec,sha)
    if old and old['controller']==c['controller'] and old['teacher_profile']==c['profile'] and old.get('deployment_sources')==sources and prefix.with_suffix('.frames.json').exists():
        check_frames(prefix,old);return old
    row=episode(spec,c['controller'],str(ck) if ck else None,prefix,capture=True,
        profile=c['profile'] if c['profile']!='v2' else 'linear',sources=sources)
    check_frames(prefix,row);return row


def replays(p,f,d,workers):
    from cooperative.replay import build
    demo=demo_plan(p);file=ROOT/'configs/deployment_demos.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8'))!=demo:raise ValueError('Demo scenario rule changed')
    atomic_json(file,demo);active=d['active_default'];tasks=[]
    for case in demo['cases']:
        s={k:case[k] for k in ['region','direction','kind','seed']}
        for c,prefix in zip([f['candidates']['nominal'],f['candidates'][active]],case['runs']):tasks.append((s,c,prefix,p['source_sha256']))
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(capture_job,tasks))
    audits=[]
    for case in demo['cases']:
        s={k:case[k] for k in ['region','direction','kind','seed']}
        for tag,prefix in zip(['nominal',active],case['runs']):
            row,audit=audit_episode(ROOT/prefix,s,f['candidates'][tag],p,'demo',tag);audits.append(audit)
            if case is demo['cases'][0]:continue
            stem=f'{s["region"]}_{s["direction"]}_{s["kind"]}_{s["seed"]}'
            reference=STUDY/'confirmation'/tag/stem
            with np.load((ROOT/prefix).with_suffix('.npz')) as a,np.load(reference.with_suffix('.npz')) as b:
                if a.files!=b.files or any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a.files):raise ValueError('Replay capture changed confirmation arrays')
    output=build(plan=demo,selected=LABELS[active],out=ROOT/'docs/통신우선_전체임무_재생_v3_1.html')
    atomic_json(STUDY/'replay_audit.json',dict(cases=3,full_missions=6,all_complete=all(r['mission_completed'] for r in results),
        matched_confirmation_runs=4,all_raw_arrays_identical=True,audits=audits,replay_sha256=sha256_file(output),interactive_browser_qa=False))
    return output


def documents(p,f,d):
    active=d['active_default'];candidate=d['selected_candidate'];m=d['confirmation_summaries'][active]['macro'];b=d['confirmation_summaries']['nominal']['macro']
    ref=d['confirmation_summaries']['v2_ppo']['macro'];best_blend=max(BLENDS,key=lambda k:tuple(f['robust_score'][k]))
    extra=d['vs_v2_ppo'].get(candidate,{}).get('search_connected_mean')
    effect='기존 PPO와 동일' if extra is None else f'{extra["difference"]*100:+.4f}%p, 공간그룹95%구간 [{extra["ci95_low"]*100:+.4f}, {extra["ci95_high"]*100:+.4f}]%p'
    table=['| 방식 | 연결률 | PDR | 총 단절(s) | 시간 배율 | 중계 이동(km) |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for k,v in d['confirmation_summaries'].items():
        r=v['macro'];table.append(f'| {LABELS[k]}'+(' **기본값**' if k==active else '')+f' | {r["search_connected_mean"]*100:.3f}% | {r["pdr"]*100:.3f}% | {r["outage_total_s"]:.1f} | {r["duration_ratio"]:.3f} | {r["relay_travel_m"]/1000:.2f} |')
    intro=f'''# 캡스톤 디자인2 · 통신 우선 학습·재배치 v3.1

현재 실행 기본값은 **{LABELS[active]} (`{active}`)**이다. 개발에서 고른 후보는 `{candidate}`, 사전 확인 승격 기준 통과는 `{d['promotion_gate_passed']}`다.

최종 확인에서 기존 비행 연결률 {b['search_connected_mean']*100:.3f}% → 기본값 {m['search_connected_mean']*100:.3f}%, PDR {b['pdr']*100:.3f}% → {m['pdr']*100:.3f}%. 같은 확인 조건의 이전 PPO는 연결률 {ref['search_connected_mean']*100:.3f}%, PDR {ref['pdr']*100:.3f}%. 후보의 이전 PPO 대비 추가 효과는 {effect}다. 구간이0을 포함하면 확실한 추가 개선으로 해석하지 않는다.

## 바로 확인

`01_통합_결과.bat` 또는 `docs/통합_결과.html`과 `docs/통합_전체임무_재생.html`을 연다. 결과·전체 임무 비교 재생은 별도 학습 없이 연다. 브라우저 파일 접근 정책으로 실제 상호작용 UI 검수는 수행하지 못했고 정적 HTML/JavaScript/프레임 데이터로 확인했다.

Python3.12/Linux CPU에서 검증했다. Windows용 BAT도 제공하며 실제 Windows 실행 검증은 별도다.

```text
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements-cooperative.txt
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python cooperate_run.py doctor
python cooperate_run.py verify
python cooperate_run.py compare --region gangnam --seed 2026 --out results/deployment/custom/gangnam
```

## 반영한 설계

- 교수님 의견대로 누적 보상으로 성능을 판단하지 않고 연결률·PDR·단절 시간·완주·실제 이동량을 비교한다.
- 단절1초=주요 비용1, 3초당 추가 이동 비용<=0.003. 긴 단절과 생성 시점별 패킷의 최종 전달/지연 품질을 포함하고 양의 생존 보상은 없다.
- 기존 Actor3개에서32개 새 전체 임무씩 실제 추가 학습했다. Critic/Adam은 새 초기화. 실제 수신 관측 기반 고정 LightGBM과 공유 PPO를 결합한다.
- 기존/새 정책은 각각의 교사점수 계약을 유지한 채 후보 확률을 혼합한다. 새 정책25/50/75%를 비교하고, 두 개발 단계 중 나쁜 쪽에서도 안정적인 후보를 고른다. 합의 결합에는 추가 역전파를 하지 않는다.
- 첨부 MADDPG의 폐루프·CTDE·지연 패킷 정산·명령 유지 아이디어를 검토했다. 미래 정답 lookahead·개발용 전역 Actor 관측·큰 사전 H 벌점은 도입하지 않았다. 관측/행동 차원이 달라 첨부 가중치를 직접 연결하지 않았다.
- 안전은 공통 이동기에서 연속5m 분리·건물/금지 구역·속도5m/s·그래프 간선으로 제한한다. 학습이 안전을 증명한 의미가 아니다.

## 최종 확인 결과

'''+ '\n'.join(table)+f'''

지도45개(학습30/초기 검증7/기존 시험8). 초기 v3 가드 방식은 검증1위였으나 v3 시험에서 이전 PPO보다 낮았다. 이 기록을 보존하고 초기 검증/시험을 새 개발 자료로 재분류했다. 최종 성능은 후보 개발에 사용하지 않은 새로운32조건×6방식=192회로 확인했다. 같은 지리8곳/6공간그룹에 새 난수를 사용한 평가이며 새 지리 영역 최초 외부 검증은 아니다.

공식 임무는 학습96 + 개발720(기존 v3의540 + 새 혼합180) + 최종 확인192 =1,008회다. 학습용 후보48회·새 시연6회·초기 가드 시연6회는 별도다. 모든 공식 궤적·링크10지표·패킷 보존·8항 보상을 검수했다. 새 학습96회와 개발/확인912회의 원시 기록·수집 시점 모델을 본 ZIP에 포함한다.

## 방식 선택과 재현

```text
python cooperate_run.py run --controller selected --region teheran --seed 2026
python cooperate_run.py run --controller consensus_ppo --region teheran --seed 2026
python cooperate_run.py run --controller priority_ppo --region teheran --seed 2026
python cooperate_run.py run --controller ppo --region teheran --seed 2026
python cooperate_run.py deployment --stage all --workers 4
python cooperate_run.py finalize --workers 4
```

`selected`는 사전 선정/확인 승격 규칙을 적용한 기본값이다. `consensus_ppo`는 개발 비교에서 고른 합의 정책 `{best_blend}`이며 기본값과 다를 수 있다. `priority_ppo`는 새 추가 학습 단독, `ppo`/`ppo_no_risk`/`mpc`는 v2 정책을 유지한다. 완료 결과는 지문 검증 후 재사용한다. 새 연구는 복사본·새 계획에서 수행한다.

## 적용 범위

- N8,수색3/중계5,V40m,400m×400m,고도50m,RF0.5초/제어3초,후보25개. 수색 드론 간 전달 목적이며 GCS/실제 조종 통신은 포함하지 않는다.
- ITU FSPL·Bullington 회절 근사·상관 섀도잉·간섭·BPSK·이상적TDMA. 전체3GPP는 아니다. 전역 이상적 라우팅·공동 안전 이동·시뮬레이션 계산 지연0 가정이 남는다.
- 실기체·Wi-Fi·배터리·GPS·바람·계산 지연 검증은 별도다. 예측기는 고정되어 공동 역전파E2E가 아니며 새 위험 제거 PPO를 따로 재학습하지 않았다.
- 공간 그룹 구간은 설명적이고 재학습 난수 불확실성은 포함하지 않는다. 후보군에서의 실측 비교이며 전역 최적 수렴을 보장하지 않는다.

새 지도는 `local-metre-scene-v1` 형식으로 `python cooperate_run.py register-map C:\\maps\\scene.local.json`을 사용한다. 동쪽x/북쪽y의 미터 좌표·400m 정사각형·건물/높이/출처·금지 구역이 필요하며 일반 경위도 GeoJSON을 직접 받지 않는다. 등록은 두 방향 안전 기본 임무를 검사하고 기존 연구 분할에는 자동 편입하지 않는다.

먼저 읽을 파일: `CONTINUE_HERE.md`, `docs/MADDPG_교수피드백_반영검토.md`, `docs/구현근거_v3_1.md`, `DATA_INDEX.md`.
'''
    (ROOT/'README.md').write_text(intro,encoding='utf-8')
    handoff=f'''# 다음 대화용 프로젝트 인계 · 최종 확인 v3.1

현재 ZIP과 이 파일을 먼저 읽는다. 오래된 긴 대화의 중단 지점이나 `results/continuity/selection.json`의 가드 후보를 현재 기본값으로 오인하지 않는다.

## 현재 확정 상태

- 현재 기본값: **{LABELS[active]} (`{active}`)**. controller=`{f['candidates'][active]['controller']}`, profile=`{f['candidates'][active]['profile']}`, checkpoint=`{f['candidates'][active]['checkpoint']}`.
- 사전 후보 `{candidate}`, 확인 승격 기준 통과 `{d['promotion_gate_passed']}`, 새 정책 승격 `{d['new_policy_promoted']}`.
- 최종 확인 연결률: 기존 비행 {b['search_connected_mean']*100:.4f}% → 기본값 {m['search_connected_mean']*100:.4f}%. PDR {b['pdr']*100:.4f}% → {m['pdr']*100:.4f}%.
- 같은 확인 조건의 이전 PPO: 연결률 {ref['search_connected_mean']*100:.4f}%, PDR {ref['pdr']*100:.4f}%. 후보 추가 효과 {effect}.
- 초기 v3는 가드가 검증1위였으나 시험에서 기존 PPO보다 낮았다. v3 시험을 새 개발 자료로 재분류했고, 새 확인 난수로 최종 평가했다. 실패를 숨기거나 v3 시험을 계속 독립 시험이라고 표시하지 않는다.
- 공식1,008회 = 추가 학습96 + 개발720 + 최종 확인192. 학습용 후보48회·새 시연6회·이전 가드 시연6회는 별도다.
- 후보 동결 SHA256 `{d['selection_sha256']}`. 확인 데이터는 후보 선택에 쓰지 않았고 사전에 정한 승격 허용/기존 정책 유지 규칙에만 사용했다.

## 주요 파일

- `results/deployment/published_selection.json`: 실제 실행 기본값. `selection.json`: 확인 전에 고른 후보/정책/모델 SHA/승격 규칙.
- `configs/deployment_plan.json`: 핵심17소스·새 확인32조건·개발 규칙. `configs/continuity_plan.json`: 원래 v3 핵심14소스. v2 핵심10소스도 변경하지 않았다.
- `results/deployment/audited_summary.json`, `confirmation_all_metrics_v3_1.csv`, `all_formal_trajectory_audit.json`.
- 새 합의 개발 원본 `results/deployment/development_validation/`84회와 `development_test/`96회. 기존9방식의 개발540회는 `results/continuity/validation/`와 `test/`를 재사용한다.
- 새 난수 최종 확인 `results/deployment/confirmation/`192회. 마지막 기본값 시연은 `results/deployment/demos/`.
- 새 원시 학습96회는 `results/continuity/training/`, 모델은 `model/continuity/ensemble.pt`와 `s28701/`,`s28702/`,`s28703/`.
- 합의 정책 `model/deployment/consensus_*.pt`는 두 부모의 상대 경로/SHA와 비중을 담은 기술자다. 실행은 `cooperative.deployment_policy.load_consensus` 사용. 일반 load_policy에 이 파일을 직접 넣지 않는다.
- `docs/통합_결과.html`, `docs/통합_전체임무_재생.html`, `docs/MADDPG_교수피드백_반영검토.md`, `docs/구현근거_v3_1.md`.
- `docs/통신우선_결과_v3.html`은 실패를 포함한 이전 v3 연구 결과다. 현재 기본값은 v3.1 게시 파일을 따른다.

## 구조와 범위

고정 LightGBM → 실제 수신 비콘/ACK 이력·알고 있는 자기 목표·정적 지도 → 후보25개 → 공유 PPO/두 교사 계약의 확률 혼합 → 공통 안전 이동. 중앙 Critic은 학습 전용이다. 기존 Actor3개에서 각32개 새 임무를 학습하고 Critic/Adam은 새 초기화했다. 학습 수집 당시 logp·모델·패킷·보상·실제 궤적을 검수했다. 합의는 고정된 두 정책의 실행 결합이며 추가 역전파는 없다.

교수님 피드백: 보상합 대신 실제 통신 지표로 선정, 단절1초=비용1,3초당 이동비용<=0.003,긴 단절·생성 시점 패킷 정산. 첨부 MADDPG는112차원/연속2D라 현재91차원 문맥/25후보와 직접 호환되지 않는다. 미래 정답lookahead·개발용 전역 Actor 관측·큰 H 벌점을 도입하지 않았다.

N8,수색3/중계5,V40m,400m×400m,고도50m,RF0.5초/제어3초. ITU/Bullington 근사·BPSK·이상적TDMA/라우팅·공동 안전 이동·계산 지연0 가정. 전체3GPP/실기체/전력/GPS/바람/지연/분산 안전 검증은 별도다. 고정 예측기라 엄밀한 공동E2E가 아니며 새 위험 제거 PPO를 따로 재학습하지 않았다. 지도8곳/공간그룹6개는 기존 지리이며 새 난수 확인이다. 구간에 재학습 난수 불확실성은 포함하지 않는다. UI는 브라우저 파일 정책으로 실제 상호작용 검수 없이 정적으로 확인했다.

## 실행

```text
python cooperate_run.py doctor
python cooperate_run.py verify
python cooperate_run.py compare --region gangnam --seed 2026 --out results/deployment/custom/gangnam
python cooperate_run.py run --controller consensus_ppo --region teheran --seed 2026
python cooperate_run.py deployment --stage all --workers 4
python cooperate_run.py finalize --workers 4
```

현재 v3.1 학습·개발·확인 원본은 본 ZIP에 있다. 더 오래된 v2 전체 원시 검수에만 이전 v2 배포 ZIP과 선택 학습3파트가 필요하다. `DATA_INDEX.md` 확인. 새 가중치/보상/선정 실험은 복사본과 별도 고정 계획에서 진행한다.
'''
    (ROOT/'CONTINUE_HERE.md').write_text(handoff,encoding='utf-8')
    v3_data_index()
    idx=(ROOT/'DATA_INDEX.md').read_text(encoding='utf-8')
    tail=idx[idx.index('## 이전 v2 학습384회:'):]
    tail=tail.replace('이전 v2 가중치·검증336회·시험288회와 감사 결과는 보존했다.', '이전 v2 가중치·검증336회/시험288회 메타데이터와 감사 결과를 보존했다. 원시 검증/시험은 이전 v2 ZIP에 있다.').replace('v2 원시 학습384회만', 'v2 원시 학습384회는')
    prefix='''# 코드·실험 원본 안내 v3.1

본 ZIP은 현재 실행·추가 학습 검수·개발·최종 확인을 재현하는 완성본이다. 이전 파일을 먼저 덮어쓸 필요가 없다.

| 경로 | 포함 내용 |
| --- | --- |
| `results/continuity/training/`, `model/continuity/` | 새 학습96회와 초기/수집 시점/최종 모델·옵티마이저·RNG |
| `results/continuity/validation/`, `test/` | 기존 v3의540회. 새 연구에서는 개발 자료로 재분류 |
| `results/deployment/development_validation/`, `development_test/` | 세 혼합 비중의 추가 개발180회 |
| `results/deployment/confirmation/` | 새 난수32조건×6방식=192회의 최종 확인 |
| `model/deployment/` | 합의 기술자3개. 실제 부모 가중치는 위 기본 모델/기존 v2 앙상블에 포함 |
| `results/deployment/*audit*.json` | 현재 공식1,008회 검수. 기존 v2의1,008회와 다른 기록이며 합산하지 않음 |
| `results/deployment/demos/` | 현재 기본값과 기존 비행의 전체 임무 시연6회 |
| `PACKAGE_CONTENTS.json` | 개별 파일 SHA256과 고정 소스/게시 결과의 연결 |

과거 v2의624개 검증/시험 원시 NPZ는 이전 v2 배포본의 버전 기록에 보존하며, 현재 ZIP에서는 공간 절약을 위해 제외한다. v2 메타데이터·모델·핵심 소스는 유지한다. 현재 v3.1 전체 검수에는 이 과거 NPZ가 필요하지 않다. 더 오래된 v2를 완전히 다시 검수할 때만 이전 v2 ZIP과 아래 선택 학습3파트를 합친다.

'''
    (ROOT/'DATA_INDEX.md').write_text(prefix+tail,encoding='utf-8')


def publish(workers=4):
    p=plan();f=verify_selection(p);d=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    if (d['source_sha256']!=p['source_sha256'] or d['selection_sha256']!=sha256_file(STUDY/'selection.json')
        or not d['formal_safety']['all_passed'] or d['formal_safety']['episodes_checked']!=1008):raise ValueError('Complete audited deployment study required')
    for name,sha in d['completed_confirmation_sha256'].items():
        if sha256_file(STUDY/'confirmation'/name)!=sha:raise ValueError('Audited confirmation aggregates changed')
    archive_v2();replay=replays(p,f,d,workers);result=report();documents(p,f,d)
    for target in ['docs/통합_전체임무_재생.html','docs/통신위험_전체임무_최종.html']:shutil.copy2(replay,ROOT/target)
    shutil.copy2(result,ROOT/'docs/통합_결과.html');shutil.copy2(ROOT/'docs/구현근거_v3_1.md',ROOT/'docs/통합_구현근거.md')
    active=d['active_default'];blend=max(BLENDS,key=lambda k:tuple(f['robust_score'][k]))
    def learned(tag):return dict(tag=tag,**f['candidates'][tag],checkpoint_sha256=f['checkpoint_sha256'][tag])
    pub=dict(format=FORMAT,selected=active,selected_candidate=f['selected_candidate'],selected_policy=f['candidates'][active],
        teacher_profile=p['teacher_profile'],checkpoint_sha256=f['checkpoint_sha256'].get(active),
        study_root='results/deployment',plan_file='configs/deployment_plan.json',evaluation_phase='confirmation',
        best_learned=learned(active),best_learned_no_risk=learned('v2_no_risk_ppo'),
        priority_ppo_checkpoint=p['priority_policy']['path'],priority_ppo_checkpoint_sha256=p['priority_policy']['sha256'],
        consensus_checkpoint=f['candidates'][blend]['checkpoint'],consensus_checkpoint_sha256=f['checkpoint_sha256'][blend],
        source_sha256=p['source_sha256'],core_source_sha256=p['base_source_sha256'],
        selection_sha256=sha256_file(STUDY/'selection.json'),audited_summary_sha256=sha256_file(STUDY/'audited_summary.json'),
        forecaster_files_sha256=d['forecaster_files_sha256'],report_sha256=sha256_file(result),replay_sha256=sha256_file(replay),
        previous_v3_test_used_as_development=True,confirmation_used_for_candidate_selection=False,
        confirmation_used_for_predeclared_promotion_gate=True,promotion_gate_passed=d['promotion_gate_passed'],
        interactive_browser_qa=False,published_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(STUDY/'published_selection.json',pub);verify_publication()
    print(json.dumps(dict(published=active,candidate=f['selected_candidate'],report=str(result),replay=str(replay)),ensure_ascii=False),flush=True)
    return pub


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    if args.workers<1:a.error('workers>=1 required')
    publish(args.workers)
