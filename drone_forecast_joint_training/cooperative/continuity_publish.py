"""검수 완료한 v3 결과·실행 기본값·전체 임무 재생을 함께 게시한다."""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch

from cooperative.analyze import METRICS
from cooperative.continuity import STUDY, protocol, verify_selection
from cooperative.continuity_run import episode
from cooperative.continuity_report import LABELS, ORDER, build as report
from cooperative.io import validated_result
from cooperative.publish import check_frames
from cooperative.refine import atomic_json
from swarm_sim.regions import ROOT, sha256_file


def demo_plan(plan):
    maps={m['key']:m for m in json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']}
    cases=[dict(region='teheran',seed=2026102801,kind='intermittent',direction='east_west')]
    for region,direction,kind in [('bucheon_jungdong_east','east_west','continuous'),
                                  ('seoul_20261023_03','north_south','intermittent')]:
        cases.append(next(s.copy() for s in plan['test'] if (s['region'],s['direction'],s['kind'])==(region,direction,kind)))
    for spec,name in zip(cases,['테헤란 · 학습 지역','부천 중동 · 시험 지역','서울 03 · 시험 지역']):
        spec.update(name=name,map=maps[spec['region']]['path'],
            info='400m×400m · 고도50m · 전체 임무 · 동일 간섭 난수 · v3 검증 선정',
            runs=[f'results/continuity/demos/{spec["region"]}_{tag}' for tag in ['nominal','selected']])
    return dict(format='communications-priority-demos-v3',validation_selected=True,
                selection_by_test_outcomes=False,case_rule='three prior demo regions and fixed direction/jammer strata; first prespecified v3 seed',cases=cases)


def capture_job(item):
    spec,candidate,prefix,sources=item
    torch.set_num_threads(1)
    prefix=ROOT/prefix
    checkpoint=str(ROOT/candidate['checkpoint']) if candidate['checkpoint'] else None
    expected=sha256_file(checkpoint) if checkpoint else None
    old=validated_result(prefix,spec,expected)
    if (old is not None and old['controller']==candidate['controller'] and old['teacher_profile']==candidate['profile']
        and old.get('continuity_sources')==sources and prefix.with_suffix('.frames.json').exists()):
        check_frames(prefix,old);return old
    row=episode(spec,candidate['controller'],checkpoint,prefix,capture=True,
                profile=candidate['profile'] if candidate['profile']!='v2' else 'linear',sources=sources)
    check_frames(prefix,row);return row


def replays(plan,freeze,workers):
    from cooperative.replay import build
    from cooperative.continuity_report import audited_episode
    demo=demo_plan(plan);path=ROOT/'configs/continuity_demos.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=demo:
        raise ValueError('Predeclared demo conditions changed')
    atomic_json(path,demo)
    tasks=[]
    for case in demo['cases']:
        spec={k:case[k] for k in ['region','direction','kind','seed']}
        for candidate,prefix in zip([plan['candidates']['nominal'],freeze['selected_policy']],case['runs']):
            tasks.append((spec,candidate,prefix,plan['source_sha256']))
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(capture_job,tasks))
    audits=[]
    for case in demo['cases']:
        spec={k:case[k] for k in ['region','direction','kind','seed']}
        for tag,prefix in zip(['nominal',freeze['selected']],case['runs']):
            _,audit=audited_episode(ROOT/prefix,spec,plan['candidates'][tag],plan['source_sha256'],'demo',tag)
            audits.append(audit)
    for case in demo['cases'][1:]:
        for tag,prefix in zip(['nominal',freeze['selected']],case['runs']):
            name=f'{case["region"]}_{case["direction"]}_{case["kind"]}_{case["seed"]}'
            original=STUDY/'test'/tag/name
            fresh=json.loads((ROOT/prefix).with_suffix('.json').read_text(encoding='utf-8'))
            previous=json.loads(original.with_suffix('.json').read_text(encoding='utf-8'))
            for metric in [*METRICS,'reward_sum','command_switch_count','relay_stationary_fraction']:
                if metric in ['forecast_p95_ms','decision_p95_ms']:continue
                if fresh[metric] is None and previous[metric] is None:continue
                if not np.isclose(fresh[metric],previous[metric],rtol=0,atol=1e-12):
                    raise ValueError('Demo recapture changed a test metric: '+metric)
            with np.load((ROOT/prefix).with_suffix('.npz')) as a,np.load(original.with_suffix('.npz')) as b:
                if a.files!=b.files or any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a.files):
                    raise ValueError('Capture changed raw test arrays')
    out=build(plan=demo,selected=LABELS[freeze['selected']],out=ROOT/'docs/통신우선_전체임무_재생_v3.html')
    atomic_json(STUDY/'replay_audit.json',dict(cases=3,full_missions=6,all_complete=all(r['mission_completed'] for r in results),
        matched_test_runs=4,capture_does_not_change_test_arrays=True,audits=audits,
        replay_sha256=sha256_file(out),interactive_browser_qa=False,
        qa_limitation='browser file access blocked; no workaround attempted; static HTML/JS/data validation only'))
    return out


def archive_v2():
    folder=ROOT/'docs/archive/continuity_v2';folder.mkdir(parents=True,exist_ok=True)
    for relative in ['README.md','CONTINUE_HERE.md','docs/통합_결과.html','docs/통합_전체임무_재생.html',
                     'docs/통신위험_전체임무_최종.html','docs/통합_구현근거.md','DATA_INDEX.md']:
        source=ROOT/relative;target=folder/source.name
        if source.exists() and not target.exists():shutil.copy2(source,target)


def data_index():
    file=ROOT/'DATA_INDEX.md';old=file.read_text(encoding='utf-8')
    marker='## 기존 예측기 재학습 자료'
    tail=old[old.index(marker):] if marker in old else ''
    text='''# 코드·실험 원본 안내 v3

본 ZIP은 이전 코드를 먼저 합칠 필요 없는 완성본이다. 지도45개·기준 이동90개·고정 예측기·기존/새 재배치 모델·새 원시 학습96회·검증252회·시험288회·학습용 후보48회·전체 임무 시연6회를 포함한다. 새 v3 전체 검수에는 별도 파일이 필요하지 않다.

## v3에 포함한 원시 기록

| 경로 | 실제 내용 |
| --- | --- |
| `results/continuity/training/` | 독립 난수3개×32개 전체 임무. 문맥·후보·마스크·선택·수집 logp·Critic 상태·보상·원시 링크·궤적·패킷 코호트 |
| `model/continuity/` | 초기 Actor/새 Critic, 모든 수집 iteration, 최종 세 정책과 앙상블. 옵티마이저·RNG도 저장 |
| `results/continuity/validation/`, `test/` | 각각252회·288회, 동일 조건9방식의 원시 NPZ·메타데이터·집계 |
| `results/continuity/pilot/`, `demos/` | 학습 전용 후보48회와 시연6회. 공식 검증/시험 통계와 구분 |
| `results/continuity/*audit*.json` | 공식636회와 후보48회·시연6회의 궤적·패킷·보상·정책 검수 |
| `PACKAGE_CONTENTS.json` | ZIP 안 모든 파일의 크기·SHA256, 고정 핵심소스와 게시 결과 연결 |

## 이전 v2 학습384회: 선택적인 원시3파트

이전 v2 가중치·검증336회·시험288회와 감사 결과는 보존했다. v2 원시 학습384회만 아래 파일에 분리돼 있다. 현재 실행·현재 v3 추가 학습 검수에는 필요하지 않으며, `python -m cooperative.refinement_analysis`로 이전 v2 전체를 다시 검수할 때만 사용한다.

- [capstone_training_s27101.zip](https://chatgpt.com/library?file_id=libfile_05419c8b1fd881919dfdbe2523100961&account_id=personal)
- [capstone_training_s27102.zip](https://chatgpt.com/library?file_id=libfile_44895320477c8191abd3e3c0c4067c4f&account_id=personal)
- [capstone_training_s27103.zip](https://chatgpt.com/library?file_id=libfile_f9e9af16eb388191852bdabea52aaeea&account_id=personal)

해당3개를 프로젝트 루트에 풀어 `results/`·`model/` 경로를 유지한다. 원본 목록과 SHA는 각 파일의 `research_evidence/training_s난수.json`에 있다. v2의1,008회와 v3의636회를 같은 새 시험처럼 합산하지 않는다. 기존 문서는 `docs/archive/continuity_v2/`에 보존했다.

'''+tail
    file.write_text(text,encoding='utf-8')


def readme(data,freeze):
    selected=freeze['selected'];m=data['summaries'][selected]['macro'];b=data['summaries']['nominal']['macro']
    best=max(ORDER,key=lambda k:data['summaries'][k]['macro']['search_connected_mean'])
    rows=['| 방식 | 연결률 | PDR | 총 단절(s) | 임무 시간 배율 | 중계 이동(km) |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k in ORDER:
        r=data['summaries'][k]['macro']
        rows.append(f'| {LABELS[k]}'+(' **검증 선정**' if k==selected else '')+f' | {r["search_connected_mean"]*100:.3f}% | {r["pdr"]*100:.3f}% | {r["outage_total_s"]:.1f} | {r["duration_ratio"]:.3f} | {r["relay_travel_m"]/1000:.2f} |')
    lines=[
        '# 캡스톤 디자인2 · 통신 우선 통합 시뮬레이터 v3','',
        f'교수님 피드백과 첨부 MADDPG 검토를 반영해, 별도 보상·제어·추가 PPO 학습·검증을 실제 실행했다. 검증 선정 기본값은 **{LABELS[selected]} (`{selected}`)**이다.',
        '',f'이번 동일 난수 시험에서 연결률 {b["search_connected_mean"]*100:.3f}% → {m["search_connected_mean"]*100:.3f}%, PDR {b["pdr"]*100:.3f}% → {m["pdr"]*100:.3f}%. 수치상 시험 최고는 {LABELS[best]} ({data["summaries"][best]["macro"]["search_connected_mean"]*100:.3f}%)이며, 시험 결과로 기본값을 바꾸지 않았다.',
        '', '## 바로 실행','',
        'Python 3.12 / Linux CPU에서 검증했다. Windows용 `00_환경설정.bat`·`01_통합_결과.bat`·`02_통합_시뮬레이션.bat`·`03_통합_검사.bat`도 제공한다. 실제 Windows 실행 검증은 별도다. 패키지 버전은 `environment_versions.txt`와 doctor 출력에 기록했다.',
        '', '```text','python -m venv .venv',
        '# Windows: .venv\\Scripts\\activate', '# Linux: source .venv/bin/activate',
        'python -m pip install -r requirements-cooperative.txt',
        'python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu',
        'python cooperate_run.py doctor','python cooperate_run.py verify',
        'python cooperate_run.py compare --region gangnam --seed 2026 --out results/continuity/custom/gangnam',
        '```','',
        '추가 실행은 `results/continuity/custom/gangnam/replay.html`을 열어 기존 비행과 전체 임무를 비교한다. 새 보상·기본 방식이 실험에 사용된다. 보고서 `docs/통합_결과.html`, 전체 임무 재생 `docs/통합_전체임무_재생.html`은 인터넷 없이 연다. 브라우저 실제 상호작용 검수는 파일 접근 정책으로 수행하지 못했고, HTML·JavaScript·프레임 데이터만 정적으로 검수했다.',
        '', '## 반영한 내용','',
        '- 단절 1초를 비용1로 두고 3초당 추가 이동 비용을 최대0.003으로 제한. 긴 단절 누적·생성 시점별 패킷 최종 전달 품질도 반영.',
        '- 누적 보상 대신 연결률·PDR·단절 시간·완주·이동량으로 선정. 모든 방식에 같은 v3 보상으로 비용 분해.',
        '- 작은 전환 비용·히스테리시스 유지 제어를 추가. 모델 기반 제어와 PPO를 동등하게 비교하고 검증 성능으로 기본값 선택.',
        '- 첨부의 미래 정답 lookahead·이상적 전역 Actor 관측·큰 사전 H 벌점은 도입하지 않았다. 첨부 가중치는 입력/행동 계약이 달라 직접 연결하지 않았다.',
        '- 기존 v2 위험 사용 Actor 3개에서 각32개 새 전체 임무로 추가 학습. Critic/Adam은 새 초기화, 예측기는 고정 LightGBM. 실제 배포 Actor 입력은 로컬 수신 관측·자기 상태·정적 지도·알고 있는 자기 목표.',
        '- `docs/MADDPG_교수피드백_반영검토.md`에 기존/새 반영 상태와 첨부 원자료 근거. v2 핵심10파일·실험·모델은 그대로 보존.',
        '', '## 시험 결과','', *rows, '',
        '학습 지도30/검증7/시험8. 학습용 후보 비교48회는 별도. 추가 학습96 + 검증252 + 새 시험288 = 공식636개 전체 임무의 연속 거리·금지 구역·그래프·링크 지표·패킷·보상을 별도 검수했다. 시연6회 중 시험 재실행4회는 원시 배열 전체가 시험과 같다.',
        '', '## 방식 지정','',
        '```text','python cooperate_run.py run --controller selected --region teheran --seed 2026',
        'python cooperate_run.py run --controller priority_ppo --region teheran --seed 2026',
        'python cooperate_run.py run --controller priority_ppo_guarded --region teheran --seed 2026',
        'python cooperate_run.py run --controller priority_mpc --region teheran --seed 2026',
        'python cooperate_run.py run --controller persistent_mpc --region teheran --seed 2026',
        'python cooperate_run.py run --controller ppo --region teheran --seed 2026','```','',
        '`selected`는 동결한 검증 기본값이다. `priority_mpc`는 학습용 후보 비교에서 고른 점수 형식, `persistent_mpc`는 학습용 비교와 동일한 선형 점수+유지 제어다. `priority_ppo`/`priority_ppo_guarded`는 새 추가 학습 앙상블이며 가드 임계값은0.15다. `mpc`/`ppo`/`ppo_no_risk`는 v2 비교 정책을 유지한다. 새 비교 실행에서는 모두 같은 v3 보상으로 기록한다.',
        '', '## 재현과 연구 기록','',
        '```text','python cooperate_run.py continuity --stage all --workers 4',
        'python cooperate_run.py finalize --workers 4',
        'python cooperate_run.py report','python cooperate_run.py replay','```','',
        '완료된 결과는 지문 확인 후 재사용한다. 새 추가 학습 원시96회·수집 시점 체크포인트·검증/시험540회·시연6회·학습용 후보48회가 ZIP에 들어 있다. 이전 v2 원시 학습384회만 별도3파트이며 현재 실행/추가 학습 검수에는 필요 없다. 이전 v2 연구 전체 재검수에만 해당 파트를 프로젝트 루트에 푼다. `DATA_INDEX.md` 확인.',
        '', '## 실제 적용 범위와 한계','',
        '- N8, 수색3/중계5, 명목 V40m,400m×400m, 고도50m. RF0.5초/이동3초, 속도5m/s·연속 분리5m·건물 금지 공통 이동기.',
        '- ITU FSPL·Bullington 회절 근사·시간 상관 섀도잉·간섭·BPSK·이상적 TDMA. 3GPP TR36.777 전체 구현은 아니다.',
        '- 수색 드론 간 전달이 목적이며 GCS 목적지·실제 조종 통신은 포함하지 않는다.',
        '- 이상적 전역 라우팅·공동 안전 이동·계산 지연 없는 시뮬레이션 시간 가정. 실기체·Wi-Fi·전력·GPS·바람·지연 검증은 별도다.',
        '- 지도는 이전 v1/v2 시험 지역을 재사용하고 난수 조건만 새로 지정했다. 공간 그룹6개의 구간은 설명적이다.',
        '- 새 보상과 지도 점수를 함께 변경했고 v3 위험 제거 PPO를 별도로 재학습하지 않았다. 보상 하나의 인과 효과·새 PPO 위험 입력만의 추가 효과·최적 수렴을 주장하지 않는다.',
        '- 예측기는 고정되어 있어 예측기와 정책까지 공동 역전파하는 엄밀한 E2E 학습은 아니다. 안전은 학습이 증명한 것이 아니라 공통 이동기 제약이다.',
        '', '## 새 지도','',
        '`local-metre-scene-v1` JSON 형식(400m 정사각형, 동쪽x/북쪽y의 미터 좌표, `region_key`, 유효 건물 `geometry_local_m`/높이/출처·비행 제한)이 필요하다. 일반 경위도 GeoJSON을 직접 받지 않는다.',
        '', '```text','python cooperate_run.py register-map C:\\maps\\scene.local.json',
        'python cooperate_run.py compare --region 새지도키 --seed 2026 --out results/continuity/custom/new_map','```','',
        '등록은 두 방향 안전 기본 임무를 확인하고 기존 학습/검증/시험 분할에는 자동 편입하지 않는다.',
    ]
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def handoff(data,freeze):
    tag=freeze['selected'];m=data['summaries'][tag]['macro'];b=data['summaries']['nominal']['macro']
    best=max(ORDER,key=lambda k:data['summaries'][k]['macro']['search_connected_mean'])
    d=data['vs_previous_mpc'].get(tag,{}).get('search_connected_mean')
    vs_previous='기존 MPC와 동일' if d is None else f'{d["difference"]*100:+.4f}%p, 공간 그룹 95%구간 [{d["ci95_low"]*100:+.4f},{d["ci95_high"]*100:+.4f}]%p'
    text=f'''# 다음 대화용 인계 · 통신 우선 v3

현재 ZIP과 이 파일을 기준으로 이어간다. 기존 긴 대화의 중단 기록과 v2 수치를 현재 결과로 오인하지 않는다.

## 완료 상태

- 교수님 보상 피드백과 첨부 MADDPG 코드·설정·체크포인트·CSV를 실제 검토했다.
- v2는 MADDPG 가중치를 직접 반영한 결과가 아니었다. v3도 미래 정답 lookahead/이상적 전역 Actor 관측/큰 H 벌점을 도입하지 않았다.
- 새 보상: 단절초 우선,3초당 이동비용<=0.003,긴 단절 추가비용,패킷 생성 코호트 최종 정산. 양의 생존 보상 없음.
- 학습용 후보48회 → 새 PPO3난수×32전체임무=96 → 검증9방식×28=252 → 새 시험9방식×32=288. 공식636회+학습용48회 검수. 시연6회 별도.
- 추가 학습은 v2 위험 사용 Actor만 초기값으로 재사용하고 Critic/Adam은 새로 초기화했다. 수집 당시 모든 행동 old logp를 해당 iteration 체크포인트로 독립 재계산했다.
- 새 기본값: **{LABELS[tag]} (`{tag}`)**, controller=`{freeze['selected_policy']['controller']}`, profile=`{freeze['selected_policy']['profile']}`, checkpoint=`{freeze['selected_policy']['checkpoint']}`.
- 동일 시험조건 연결률 {b['search_connected_mean']*100:.4f}% → {m['search_connected_mean']*100:.4f}%, PDR {b['pdr']*100:.4f}% → {m['pdr']*100:.4f}%.
- 기존 MPC와 대응 비교: {vs_previous}.
- 시험 평균 수치상 최고는 {LABELS[best]} {data['summaries'][best]['macro']['search_connected_mean']*100:.4f}%. 시험 결과로 기본값을 다시 선택하지 않았다.
- 검증 선정 SHA256 `{data['selection_sha256']}`.

## 먼저 읽을 파일

- `README.md`, `docs/통합_결과.html`, `docs/통합_전체임무_재생.html`.
- `docs/MADDPG_교수피드백_반영검토.md`, `docs/구현근거_v3.md`.
- `configs/continuity_plan.json`: 실제 추가 학습/검증/시험 조건과 핵심14소스 지문.
- `results/continuity/selection.json`: 시험 전에 고른 기본값.
- `results/continuity/published_selection.json`: 실제 CLI 기본값과 체크포인트.
- `results/continuity/audited_summary.json`, `comparison_all_metrics_v3.csv`.
- `results/continuity/all_formal_trajectory_audit.json`: 공식636회, 원시 링크10지표·연속 거리·금지구역·패킷·명령·보상 독립검수.
- `model/continuity/ensemble.pt`, `model/continuity/s28701,s28702,s28703/`: 새 정책과 모든 수집 시점 모델.
- `results/continuity/training/`: 새 원시 학습96회가 본 ZIP에 포함된다. 이전 v2의384회 원본만 별도3파트.
- `docs/archive/continuity_v2/`, `results/refinement/`: v2 결과 보존. v2 핵심10파일은 수정하지 않았다.

## 실행

```text
python cooperate_run.py doctor
python cooperate_run.py verify
python cooperate_run.py compare --region gangnam --seed 2026 --out results/continuity/custom/gangnam
python cooperate_run.py run --controller priority_ppo --region teheran --seed 2026
python cooperate_run.py continuity --stage all --workers 4
python cooperate_run.py finalize --workers 4
```

`selected`는 검증 기본값, `priority_ppo`는 새 학습, `persistent_mpc`는 선형 점수+유지 제어, `priority_mpc`는 학습용 후보 비교에서 선정한 점수다. 기존 `ppo`/`ppo_no_risk`/`mpc`는 v2 비교 방식을 유지한다. 완료 기록은 지문 검증 후 재사용한다. 새 연구는 복사본과 별도 계획에서 수행한다.

## 주장 범위

- N8,수색3/중계5,V40m,400m×400m,고도50m,RF0.5초/제어3초,25후보. 고정 LightGBM+공유 PPO 잔차+학습 전용 중앙 Critic.
- 실제 수신 관측과 알고 있는 자기 목표/정적 지도만 Actor에 공급한다. 미래 정답 간섭/위치 입력 없음. 이상적 전역 라우팅·공동 안전 이동은 여전히 시뮬레이터 가정이다.
- 예측기-정책 공동 역전파 E2E, 전체3GPP,실기체/전력/GPS/바람/계산 지연 검증은 포함하지 않는다.
- 지도45개(학습30/검증7/시험8), 시험은 기존 지도에 새 난수이며 독립 공간 그룹6개. 새 지리 영역 최초 외부검증이 아니다.
- 새 보상·점수를 함께 변경해 보상만의 인과 효과는 확정하지 못한다. 새 위험 제거 PPO를 별도 재학습하지 않아 새 PPO 위험 입력만의 추가효과를 주장하지 않는다.
- UI는 파일 접근 정책 때문에 실제 브라우저 상호작용 검수를 못 했다. 정책 우회 없이 정적 HTML/JavaScript/프레임 데이터 검수와 과학 그림 확인을 수행한다.
'''
    (ROOT/'CONTINUE_HERE.md').write_text(text,encoding='utf-8')


def publish(workers=4):
    plan=protocol();freeze=verify_selection(plan)
    data=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    if (data['selection_sha256']!=sha256_file(STUDY/'selection.json') or data['source_sha256']!=plan['source_sha256']
        or data['selected_by_validation']!=freeze['selected'] or data['counts']!=dict(training=96,validation=252,test=288,formal=636,pilot=48)
        or not data['formal_safety']['all_passed'] or not data['pilot_safety']['all_passed']
        or not data['reward_audit']['all_reconstructed']):raise ValueError('Complete audited v3 study required')
    for name,sha in data['completed_test_sha256'].items():
        if sha256_file(STUDY/'test'/name)!=sha:raise ValueError('Audited test aggregates changed')
    archive_v2();replay=replays(plan,freeze,workers);result=report();readme(data,freeze);handoff(data,freeze);data_index()
    shutil.copy2(result,ROOT/'docs/통합_결과.html');shutil.copy2(replay,ROOT/'docs/통합_전체임무_재생.html')
    shutil.copy2(replay,ROOT/'docs/통신위험_전체임무_최종.html')
    shutil.copy2(ROOT/'docs/구현근거_v3.md',ROOT/'docs/통합_구현근거.md')
    def best_learned(tags):
        feasible=[k for k in tags if freeze['feasible'][k]]
        tag=min(feasible or tags,key=lambda k:(-freeze['summaries'][k]['macro']['search_connected_mean'],
                    -freeze['summaries'][k]['macro']['timely_pdr'],freeze['summaries'][k]['macro']['relay_travel_m']))
        return dict(tag=tag,**freeze['candidates'][tag],checkpoint_sha256=freeze['checkpoint_sha256'][tag])
    manifest=dict(format='communications-priority-published-v3',selected=freeze['selected'],selected_policy=freeze['selected_policy'],
        teacher_profile=plan['teacher_profile'],checkpoint_sha256=freeze['checkpoint_sha256'].get(freeze['selected']),
        priority_ppo_checkpoint='model/continuity/ensemble.pt',priority_ppo_checkpoint_sha256=sha256_file(ROOT/'model/continuity/ensemble.pt'),
        best_learned=best_learned(['v2_ppo','priority_ppo','priority_ppo_guarded']),
        best_learned_no_risk=best_learned(['v2_no_risk_ppo']),
        validation_selection_sha256=sha256_file(STUDY/'selection.json'),audited_summary_sha256=sha256_file(STUDY/'audited_summary.json'),
        report_sha256=sha256_file(result),replay_sha256=sha256_file(replay),source_sha256=plan['source_sha256'],
        forecaster_files_sha256=plan['forecaster_files'],test_used_for_selection=False,
        interactive_browser_qa=False,published_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(STUDY/'published_selection.json',manifest)
    print(json.dumps(dict(published=freeze['selected'],report=str(result),replay=str(replay)),ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.workers<1:p.error('workers>=1 required')
    publish(a.workers)
