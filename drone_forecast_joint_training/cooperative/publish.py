"""검증 선정·원시 시험 검수 이후 새 결과와 전체 임무 재생을 게시."""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import torch

from cooperative.analyze import METRICS
from cooperative.io import validated_result
from cooperative.refine import STUDY, atomic_json, protocol
from cooperative.refinement_report import LABELS, build as report
from cooperative.run import episode
from cooperative.study import validate_freeze
from swarm_sim.regions import ROOT, sha256_file


def demo_plan(plan):
    maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    by_key={m['key']:m for m in maps}
    cases=[dict(region='teheran',seed=2026102601,kind='intermittent',direction='east_west')]
    for region,direction,kind in [('bucheon_jungdong_east','east_west','continuous'),
                                  ('seoul_20261023_03','north_south','intermittent')]:
        cases.append(next(s.copy() for s in plan['test'][::2] if (s['region'],s['direction'],s['kind'])==(region,direction,kind)))
    names=['테헤란 · 학습 지역','부천 중동 · 시험 지역','서울 03 · 시험 지역']
    for spec,name in zip(cases,names):
        spec.update(name=name,map=by_key[spec['region']]['path'],
                    info='400m×400m · 고도 50m · 새 실행의 전체 임무 · 공통 간섭 난수',
                    runs=[f'results/refinement/demos/{spec["region"]}_{tag}' for tag in ['nominal','selected']])
    return dict(format='cooperative-refinement-demos-v2',selection_by_test_outcomes=False,
                case_rule='same three v1 regions and direction/jammer strata; first predeclared v2 test seed for test regions',cases=cases)


def check_frames(prefix,summary):
    data=json.loads(prefix.with_suffix('.frames.json').read_text(encoding='utf-8'))
    frames=data['frames']
    with np.load(prefix.with_suffix('.npz')) as z:
        if not (summary['mission_completed'] and frames[0]['time_s']==0
                and frames[-1]['time_s']==summary['duration_s']
                and len(frames)==len(z['nodes'])==len(z['C'])//6+1):
            raise ValueError('전체 임무 재생 프레임 불일치: '+str(prefix))
    return len(frames)


def capture_job(item):
    spec,controller,checkpoint,prefix,sources=item
    torch.set_num_threads(1)
    checkpoint=str(ROOT/checkpoint) if checkpoint else None
    prefix=ROOT/prefix;expected=sha256_file(checkpoint) if checkpoint else None
    old=validated_result(prefix,spec,expected)
    if old is not None and old.get('controller')==controller and old.get('refinement_sources')==sources and prefix.with_suffix('.frames.json').exists():
        check_frames(prefix,old)
        return old
    result=episode(spec,controller,checkpoint,prefix,capture=True)
    result['refinement_sources']=sources
    atomic_json(prefix.with_suffix('.json'),result)
    check_frames(prefix,result)
    return result


def replays(plan,freeze,workers):
    from cooperative.replay import build
    demo=demo_plan(plan)
    path=ROOT/'configs/refinement_demos.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=demo:
        raise ValueError('사전에 지정한 시연 조건 변경')
    atomic_json(path,demo)
    candidate=freeze['selected_policy'];tasks=[]
    for case in demo['cases']:
        spec={k:case[k] for k in ['region','direction','kind','seed']}
        tasks.extend([(spec,'nominal',None,case['runs'][0],plan['source_sha256']),
                      (spec,candidate['controller'],candidate['checkpoint'],case['runs'][1],plan['source_sha256'])])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(capture_job,tasks))
    for case in demo['cases'][1:]:
        for tag,prefix in zip(['nominal',freeze['selected']],case['runs']):
            name=f'{case["region"]}_{case["direction"]}_{case["kind"]}_{case["seed"]}'
            original=STUDY/'test'/tag/name
            fresh=json.loads((ROOT/prefix).with_suffix('.json').read_text(encoding='utf-8'))
            expected=json.loads(original.with_suffix('.json').read_text(encoding='utf-8'))
            for metric in METRICS:
                if metric in ['forecast_p95_ms','decision_p95_ms']:continue
                if fresh[metric] is None and expected[metric] is None:continue
                if not np.isclose(fresh[metric],expected[metric],atol=1e-12,rtol=0):
                    raise ValueError('시험 결과와 시연 재실행의 지표 불일치: '+metric)
            with np.load((ROOT/prefix).with_suffix('.npz')) as a,np.load(original.with_suffix('.npz')) as b:
                if a.files!=b.files or any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a.files):
                    raise ValueError('시연을 기록하는 실행에서 원시 시험 배열 변경')
    output=build(plan=demo,selected=LABELS[freeze['selected']],out=ROOT/'docs/통합_전체임무_재생_v2.html')
    atomic_json(STUDY/'replay_audit.json',dict(cases=3,full_missions=len(results),
                all_complete=all(r['mission_completed'] for r in results),matched_test_runs=4,
                capture_does_not_change_test_arrays=True,replay_sha256=sha256_file(output)))
    return output


def archive_v1():
    folder=ROOT/'docs/archive/cooperative_v1';folder.mkdir(parents=True,exist_ok=True)
    for relative in ['README.md','docs/통합_결과.html','docs/통합_전체임무_재생.html','docs/통신위험_전체임무_최종.html','docs/통합_구현근거.md']:
        original=ROOT/relative;target=folder/original.name
        if original.exists() and not target.exists():shutil.copy2(original,target)


def readme(data,freeze):
    best=max(data['summaries'],key=lambda tag:data['summaries'][tag]['macro']['search_connected_mean'])
    best_policy=freeze['candidates'][best]
    command='python cooperate_run.py run --controller '+best_policy['controller']
    if best_policy['checkpoint']:command+=' --checkpoint '+best_policy['checkpoint']
    effect=data['matched_forecast_ablation']['risk_vs_no_risk']['search_connected_mean']
    detail=(f"이번 시험 평균 연결률의 수치상 최고는 **{LABELS[best]} ({data['summaries'][best]['macro']['search_connected_mean']*100:.3f}%)**. "
            f"해당 방식의 추가 실행: `{command}`. 기본값은 시험 전에 검증에서 선정한 방식을 유지한다.\n\n"
            f"같은 PPO 구조에서 예측 위험 추가 효과는 {effect['difference']*100:+.3f}%p, 공간 그룹 95% 구간 "
            f"[{effect['ci95_low']*100:+.3f}, {effect['ci95_high']*100:+.3f}]%p. 예측 위험의 추가 이득은 해당 구간과 대조군 결과로 판단한다.")
    selected=freeze['selected'];metrics=data['summaries'][selected]['macro']
    baseline=data['summaries']['nominal']['macro'];rows=[]
    for tag in ['nominal','reactive','mpc_no_risk','mpc','legacy_ppo','no_risk_ensemble','risk_ensemble','no_risk_ensemble_guarded','risk_ensemble_guarded']:
        item=data['summaries'][tag];m=item['macro']
        rows.append(f'| {LABELS[tag]}'+(' **검증 선정**' if tag==selected else '')+f' | {m["search_connected_mean"]*100:.2f}% | {m["pdr"]*100:.2f}% | {item["complete_count"]}/32 |')
    text='''# 군집 UAV 통신 예측·학습·재배치 통합 시뮬레이터 v2

실제 도시 지도의 비행·무선 채널·패킷 송수신을 계산하고, 수신 관측의 예측 위험으로 중계 드론을 재배치하는 폐루프 구현.

**확인 순서: `01_통합_결과.bat` → 결과 설명서 → 세 지역 전체 임무 비교 재생.** HTML 열람에 Python 설치나 재학습은 필요하지 않다.

검증에서 선정한 기본 방식은 **__SELECTED__**. 새 시험 32조건의 연결 유지율은 **__CTR__%**, 패킷 전달률은 **__PDR__%**이며 기존 비행 대비 각각 **__DC__%p**, **__DP__%p**. PPO와 MPC의 차이·예측 위험의 기여·공간 그룹 불확실성은 결과 설명서에 함께 표시한다. 모든 비교군에서 최고라는 결론이나 실기체 검증 결과를 뜻하지 않는다.

## 새 시험 결과

__TEST_DETAIL__

| 방식 | 연결 유지율 | 패킷 전달률 | 전체 임무 완주 |
|---|---:|---:|---:|
__ROWS__

시험 지도 8곳 × 2방향 × 2간섭 조건. 학습 30 / 검증 7 / 시험 8 지도 분할을 유지했다. 기존 v1에서 보았던 시험 지도에 새로운 난수 조건을 적용한 재평가이며 새로 수집한 도시의 최초 시험은 아니다. 서로 겹치는 시험 지도는 동일 공간 그룹으로 묶어 독립 그룹 6개로 대응 구간을 계산한다. 기존 v1 수치는 `docs/archive/cooperative_v1/`에 별도 보존한다.

## 주요 파일

| 경로 | 역할 |
|---|---|
| `docs/통합_결과.html` | 실제 새 시험 결과·선정 근거·예측 기여·검수·한계 |
| `docs/통합_전체임무_재생.html` | 세 지역의 처음부터 마지막까지 동기화 비교 |
| `docs/통합_구현근거.md` | 구조·입력 계약·학습·안전·문헌과 설계 선택의 구분 |
| `cooperate_run.py` | 환경 검사·새 실행·비교·보고서 진입점 |
| `cooperative/study.py` | 경로와 무관한 전체 연구 재현·중단 재개 |
| `model/advance/selected/best.pt` | 고정 LightGBM 예측기 |
| `model/refinement/` | 예측 유무별 3개 학습 난수, 반복별 체크포인트, 앙상블 |
| `results/refinement/published_selection.json` | 실행용 검증 선정·모델 지문 |
| `results/refinement/selection.json` | 시험 전에 동결한 선정·검증 수치·조건 |
| `results/refinement/audited_summary.json` | 새 시험 집계·대응 구간·학습 검수 |
| `results/refinement/comparison_all_metrics_v2.csv` | 실행별 전체 지표·패킷 수·이동량 |
| `results/refinement/trajectory_audit.json` | 288회 시험 궤적의 독립 안전 검수 |
| `results/refinement/all_formal_trajectory_audit.json` | 학습·검증·시험 1,008회 전체 독립 궤적·링크 검수 |
| `configs/refinement_plan.json` | 고정 코드·예측기 지문·학습/검증/시험 계획 |
| `configs/refinement_execution.json` | 사전 지정 예산·32조건 부분집합·후보 수 |
| `DATA_INDEX.md` | 원시 학습 증거·예측 데이터·과거 결과 위치 |

## Windows 실행

1. Python 3.12 설치 후 ZIP 전체 해제. 예: `C:\\capstone`.
2. `00_환경설정.bat`: 가상환경, 검증한 CPU 의존성, PyTorch 설치와 검사.
3. `01_통합_결과.bat`: 결과·전체 임무 재생 열기.
4. `02_통합_시뮬레이션.bat`: 새 조건의 기존 비행/검증 선정 방식 비교.
5. `03_통합_검사.bat`: 지도·가중치·패킷·회귀 검사.

계산·자동 검수는 Linux CPU/Python 3.12에서 수행했다. Windows BAT 제공과 실제 Windows PC 실행 검증은 구분한다. 운영체제 변경·ns-3 설치는 필요하지 않다.

```text
.venv\\Scripts\\python.exe cooperate_run.py doctor
.venv\\Scripts\\python.exe cooperate_run.py compare --region gangnam --seed 2026 --out results/refinement/custom/gangnam
.venv\\Scripts\\python.exe cooperate_run.py compare --region gangnam --seed 2026 --controller ppo --out results/refinement/custom/gangnam_ppo
.venv\\Scripts\\python.exe cooperate_run.py run --region bundang --controller ppo --out results/refinement/custom/bundang_ppo
.venv\\Scripts\\python.exe cooperate_run.py run --region bundang --controller ppo_no_risk --out results/refinement/custom/bundang_no_risk
.venv\\Scripts\\python.exe cooperate_run.py run --region hongdae --controller nominal --out results/refinement/custom/hongdae
```

기본은 검증 선정 방식이다. `ppo`와 `ppo_no_risk`는 각각 검증에서 선정한 위험 유무별 학습 방식과 가드를 사용한다. 가드 명시 실행은 `ppo_guarded --checkpoint 모델경로`로 지정한다. `nominal`, `reactive`, `mpc`, `mpc_no_risk`도 선택할 수 있다. `compare`는 두 방식의 전체 임무를 새로 실행하고 출력 폴더에 JSON·NPZ·프레임·`replay.html`을 저장한다. 공식 시험 집계·선정은 자동 변경하지 않는다.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validated.txt
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python cooperate_run.py doctor
python cooperate_run.py verify
```

## 학습과 평가 재현

새 정책은 위험 유무별 독립 학습 난수 3개, 정책당 전체 임무 64회, 총 384회로 학습했다. 개별 학습 난수의 14조건 진단은 최선 난수 선택에 사용하지 않는다. 동일 가중치의 3개 정책 logit 평균 앙상블과 가드 유무를 28조건에서 검증한다. 검증/진단 총 336회, 새 시험은 9방식 × 32조건 = 288회다.

```text
python -m cooperative.study --stage all --workers 4
python -m cooperative.study --stage validation --workers 4
python -m cooperative.study --stage test --workers 4
python -m cooperative.study --stage report
python -m cooperative.publish --workers 4
```

전체 재현은 CPU에서 오래 걸린다. 완료된 체크포인트·실행 파일은 난수·지문·배열 무결성을 검사해 재사용한다. 원시 학습 NPZ를 분리 제공한 경우 `DATA_INDEX.md`의 증거 ZIP을 프로젝트에 먼저 풀어야 전체 학습 검수가 가능하다. 기본 실행과 HTML 열람에는 원시 학습 NPZ가 필요하지 않다.

학습 수집 직후 잔차 크기를 외부 변경하던 문제를 수정했고, 매 PPO 갱신 전에 모든 수집 행동의 log-probability를 다시 비교한다. 위험 제거군은 Actor·Critic 위험 입력을 모두 제거하고 예측기를 실행하지 않는다. 비콘 비용·기하학적 속도 외삽은 동일하다.

새 연구·다른 보상·더 긴 학습은 프로젝트 복사본에서 별도 결과 이름과 계획으로 관리해야 한다. 배포 모델·선정·시험 조건을 덮어쓰며 시험 결과로 조정하는 흐름을 피한다. 기존 v1 학습은 현재 수집 계약과 달라 비교용 모델로 보존한다.

## 고정 설계·실제 적용 범위

- 전체 8대, 수색 3대·중계 5대, 명목 V 대형 40m.
- 400m×400m, 고도 50m, 비행 격자 10m, 연속 이동 최소 분리 5m.
- 관측 0.5초, 예측·이동 선택 3초, 이상적 TDMA 5ms.
- 25개 로컬 이동 후보, 지도 점수 + 공유 PPO 잔차, 학습 전용 중앙 Critic.
- 고정 LightGBM 예측기. 예측기까지 함께 역전파하는 공동 E2E 학습은 포함하지 않는다.
- 수색 드론 간 전달 목적이며 GCS 목적지나 실제 기체 연결은 포함하지 않는다.
- ITU FSPL·Bullington 지붕 회절 근사·시간 상관 섀도잉·외부 간섭·BPSK 데이터/ACK. 3GPP TR 36.777 전체 구현은 아니다.
- 이상적 전역 라우팅·공동 안전 이동·계산 지연 없는 시뮬레이션 시간 가정. 실제 Wi-Fi 스택·기체·에너지·위치 오차·바람과의 통합 검증은 별도다.

가드는 PPO 행동의 로컬 모델 점수가 최선 후보보다 0.35 초과 낮으면 모델 기반 후보를 선택한다. 실제 미래 채널을 사용하지 않는다. 전달률 저하 0.5%p 이내·임무 시간 최대 1.15배는 프로젝트 선정 제한이다. 안전 보호는 학습이 보장한 결과가 아니라 공통 이동기의 제약이다.

## 새 지도 등록

`local-metre-scene-v1` JSON: 중심 원점의 동쪽 x·북쪽 y(m), `local_bounds: [-200,-200,200,200]`, 고유 영문 `region_key`, 유효한 `geometry_local_m`, 건물 높이·출처, 비행 제한 영역이 필요하다. 일반 경위도 GeoJSON을 직접 넣는 형식은 아니다.

```text
python cooperate_run.py register-map C:\\maps\\scene.local.json
python cooperate_run.py compare --region 새지도키 --seed 2026 --out results/refinement/custom/new_map
```

등록 시 두 방향의 기본 안전 경로와 전체 수색 완주를 확인한다. 새 지도는 기존 학습/검증/시험 분할에 자동 편입하지 않는다.
'''
    replacements={'__TEST_DETAIL__':detail,'__SELECTED__':LABELS[selected],'__CTR__':f'{metrics["search_connected_mean"]*100:.2f}',
                  '__PDR__':f'{metrics["pdr"]*100:.2f}','__DC__':f'{(metrics["search_connected_mean"]-baseline["search_connected_mean"])*100:+.3f}',
                  '__DP__':f'{(metrics["pdr"]-baseline["pdr"])*100:+.3f}','__ROWS__':'\n'.join(rows)}
    for a,b in replacements.items():text=text.replace(a,b)
    (ROOT/'README.md').write_text(text,encoding='utf-8')


def handoff(data,freeze):
    tag=freeze['selected'];m=data['summaries'][tag]['macro'];b=data['summaries']['nominal']['macro']
    best=max(data['summaries'],key=lambda name:data['summaries'][name]['macro']['search_connected_mean'])
    effect=data['matched_forecast_ablation']['risk_vs_no_risk']['search_connected_mean']
    lines=[
        '# 다음 대화용 프로젝트 인계 v2',
        '',
        '기존 긴 대화를 다시 읽는 대신 이 파일과 ZIP을 먼저 확인한다. 아래 상태는 새 실제 실행 결과 기준이며 과거 대화의 v1 수치와 별도다.',
        '',
        '## 현재 결과',
        '',
        f'- 검증 선정 기본 방식: {LABELS[tag]} (`{tag}`), controller=`{freeze["selected_policy"]["controller"]}`.',
        f'- 기본 체크포인트: `{freeze["selected_policy"]["checkpoint"]}`. None이면 모델 기반 방식.',
        f'- 새 시험 수색군 연결률: {b["search_connected_mean"]*100:.4f}% → {m["search_connected_mean"]*100:.4f}%.',
        f'- 새 시험 패킷 전달률: {b["pdr"]*100:.4f}% → {m["pdr"]*100:.4f}%.',
        f'- 이번 시험 평균 연결률 수치상 최고: {LABELS[best]} (`{best}`), {data["summaries"][best]["macro"]["search_connected_mean"]*100:.4f}%. 시험 전 검증 기본값과 구분한다.',
        f'- 같은 PPO 구조의 예측 위험 추가 효과: {effect["difference"]*100:+.3f}%p, 공간 그룹 95% 구간 [{effect["ci95_low"]*100:+.3f}, {effect["ci95_high"]*100:+.3f}]%p. 추가 이득을 자동으로 전제하지 않는다.',
        f'- 학습 위험 사용 최선 학습 방식: `{freeze["best_learned_risk"]}`.',
        f'- 학습 위험 제거 최선 학습 방식: `{freeze["best_learned_no_risk"]}`.',
        '- 공식 전체 임무: 학습 384 + 검증/독립 난수 진단 336 + 새 시험 288 = 1,008회. 시연 6회는 별도.',
        '- 지도 45개: 학습30 / 검증7 / 시험8. 시험은 기존 지도에 새 난수32조건, 독립 공간 그룹6개.',
        '- 기본 방식은 시험 전에 선정했다. 시험 결과를 보고 기본값이나 보상을 다시 조정하지 않는다.',
        '',
        '## 구현과 수정',
        '',
        '- 고정 LightGBM 예측기 → 로컬 후보25개 → 지도 점수 + 공유 PPO 잔차 → 선택적 가드 → 공통 안전 이동.',
        '- 위험 유무별 독립 학습 난수27101/27102/27103, 정책당64개 전체 임무, 세 정책 logit 평균 앙상블.',
        '- 수집 직후 잔차 크기를 외부 변경하던 순서 문제 수정. 매 갱신 전에 전체 행동 old logp 오차<=2e-5 검사.',
        '- 위험 제거군을 별도로 처음부터 학습. Actor/Critic 위험 특징 제거, 예측기 호출0, 비콘 비용·속도 외삽 유지.',
        '- 가드: 선택 행동의 로컬 모델 점수가 최선보다0.35 초과 낮으면 모델 후보로 복귀. 미래 정답 입력 없음.',
        '- RF는 정확한 좌표·시나리오 캐시. 원래 RF 구현을 tests/fixtures에 보존해 결과 직접 비교.',
        '- 공식1,008회 저장 궤적의 연속 최소 거리·금지 구역·그래프 간선·패킷 보존과 링크 행렬의 전체10개 지표를 별도 검수.',
        '',
        '## 실제 범위',
        '',
        '- N8, 수색3/중계5,400x400m,고도50m,V40m,분리5m. 관측0.5초/제어3초.',
        '- 예측기는 고정하며 예측기-정책 공동 역전파 E2E 학습은 없다.',
        '- ITU/Bullington 근사·BPSK·이상적 TDMA/라우팅·공동 안전 이동. 실제 기체/바람/GPS오차/배터리/계산 지연 검증은 별도.',
        '- MPC·반응형·이전 PPO도 같은 새 조건에서 비교했다. 모든 조건의 PPO 우위를 전제하지 않는다.',
        '',
        '## 먼저 읽을 파일',
        '',
        '- `README.md`, `docs/통합_결과.html`, `docs/통합_전체임무_재생.html`.',
        '- `results/refinement/selection.json`: 시험 전 동결 검증 선정.',
        '- `results/refinement/published_selection.json`: 실제 CLI 기본 방식과 모델 SHA.',
        '- `results/refinement/audited_summary.json`: 수치·공간 그룹 구간·학습 검수.',
        '- `results/refinement/comparison_all_metrics_v2.csv`, `trajectory_audit.json`.',
        '- `results/refinement/verification/`: 회귀 검사, 배포 폴더 검사, 원래 RF/전체 임무 동등성.',
        '- `configs/refinement_plan.json`, `refinement_execution.json`: 고정 연구 계획·핵심 코드 지문.',
        '- `docs/구현근거_v2.md`: 모듈과 입력 계약. `DATA_INDEX.md`: 분리된 학습 원본3파트.',
        '',
        '## 실행과 이어서 연구',
        '',
        '```text',
        'python cooperate_run.py doctor',
        'python cooperate_run.py verify',
        'python cooperate_run.py compare --region gangnam --seed 2026 --out results/refinement/custom/gangnam',
        'python -m cooperative.study --stage all --workers 4',
        '```',
        '',
        '완료 파일은 지문 검증 후 재사용한다. 전체 학습 검수에는 별도 capstone_training_s27101/27102/27103.zip을 프로젝트 루트에 풀어야 한다. 기본 시뮬레이션과 HTML에는 필요하지 않다. 새 보상·더 긴 학습·새 선정은 프로젝트 복사본에서 별도 계획으로 진행한다. 현재 기본 결과와 시험 조건을 덮어쓰지 않는다.',
        '',
    ]
    (ROOT/'CONTINUE_HERE.md').write_text('\n'.join(lines),encoding='utf-8')


def publish(workers=4):
    plan=protocol();freeze=validate_freeze(json.loads((STUDY/'selection.json').read_text(encoding='utf-8')),plan)
    data=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    if (data['selected_by_validation']!=freeze['selected'] or data['test_episodes']!=288
            or not data['safety']['all_passed'] or not data['formal_safety']['all_passed']
            or data['formal_safety']['episodes_checked']!=1008 or not all(r['all_complete'] for r in data['training_audit'])):
        raise ValueError('완료된 독립 시험·학습 검수가 필요')
    archive_v1();replay=replays(plan,freeze,workers);result=report();readme(data,freeze);handoff(data,freeze)
    shutil.copy2(result,ROOT/'docs/통합_결과.html');shutil.copy2(replay,ROOT/'docs/통합_전체임무_재생.html')
    shutil.copy2(replay,ROOT/'docs/통신위험_전체임무_최종.html')
    shutil.copy2(ROOT/'docs/구현근거_v2.md',ROOT/'docs/통합_구현근거.md')
    def learned(tag):
        return dict(tag=tag,**freeze['candidates'][tag],checkpoint_sha256=freeze['checkpoint_sha256'][tag])
    selected=freeze['selected']
    manifest=dict(format='cooperative-published-v2',selected=selected,selected_policy=freeze['selected_policy'],
                  checkpoint_sha256=freeze['checkpoint_sha256'].get(selected),
                  best_learned=learned(freeze['best_learned_risk']),
                  best_learned_no_risk=learned(freeze['best_learned_no_risk']),
                  validation_selection_sha256=sha256_file(STUDY/'selection.json'),
                  audited_summary_sha256=sha256_file(STUDY/'audited_summary.json'),
                  report_sha256=sha256_file(result),replay_sha256=sha256_file(replay),
                  forecaster_sha256=plan['forecaster_sha256'],forecaster_files_sha256=data['forecaster_files_sha256'],source_sha256=plan['source_sha256'],
                  test_used_for_selection=False,published_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(STUDY/'published_selection.json',manifest)
    print(json.dumps({'published':selected,'report':str(result),'replay':str(replay)},ensure_ascii=False))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers는 1 이상 필요')
    publish(args.workers)
