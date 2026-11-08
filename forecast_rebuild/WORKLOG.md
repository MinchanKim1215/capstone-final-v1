# 재구성 작업 진행 기록

기준: 2026-11-02 01:30 KST 전후. 사용자는 기존을 보존하고 예측 성능을 위해 새 구성·학습·비교까지 적극 수행하도록 승인했다. 실제 기체·통신모듈은 미정이다.

## 완료

- v5/v6/v7 원본 코드·모델·최종 평가 지도 32개 보존. `contracts.verify_prior()` 검사.
- 새 폴더 `forecast_rebuild`, 버전 `forecast-rebuild-v8-20261102`.
- 목표: 모든 상태의 지속 단절 시작을 1/3/5/8초 및 3~4/3~7초 창으로 지도학습. 기존 현재 연결 링크의 다음 5초 단절과 같은 정답으로 직접 비교도 유지.
- 미래 관측·미수신 메시지·시간 경계 검사 8개 통과. 실제 GPU 역전파·LightGBM 네이티브 학습 확인.
- 새 고정 v7 64회 정책 비행 139개: 새 학습 지도 79 + 기존 학습 지도 16 + 개발 검증 지도 44. 모든 파일 해시 확인.
- 수집 기록 작성의 `array_path` 인터페이스 오류와 Windows/OneDrive 파일 교체 충돌을 수정. RF 비행 자체는 모두 저장됐으며 `recover_collection.py`가 검증해 목록 복원. 현재 collection_status는 completed.
- 전체 훈련 기록 319비행(기존 bootstrap192 + v6/v7 마지막32 + 새 학습95), 학습 지도 245개 전부 포함. 선택 24비행, 보정 20비행 준비. 확인 16비행(새14 + 과거2)은 모델·경보 동결 전 읽지 않음.
- 선택적 유휴 전파 에너지 관측: 현재 잡음+간섭을 0.75dB 잡음·0.1dB 양자화·1% 결측으로 모사. 이웃 값은 성공 수신 비콘의 이전 측정만 전달. 실제 간섭 좌표·출력·스케줄은 모델에 전달하지 않음. 현재 RF 세계는 그대로. 하드웨어 지원 미확인.
- 센서 물리 측정 시각: own은 0.5초 전, 수신 peer의 이전 측정은 최소1초 전. 버전s2 캐시 사용. 초기 s1 캐시는 engineering 실험이 센서 없이 사용한 자료만 보존.
- 지도+과거 수신 위치로 명목 경로 진행을 추정하는 `planning.py`와 등속도 대안, 신뢰도 82개 추가 특징 구현. 실제 미래 위치 미사용. 첫 두 지도 진단은 혼합 결과이므로 향상 주장 없음.
- 실시간 로컬 API `runtime.LocalForecast` 구현. 실제 자료에서 배치/스트림 확률 오차0, 계획 특징 오차1.86e-9. 결측 위험은 None, 무검증 자동 경보는 false. CPU 첫 호출319ms, 이후 약3ms(데스크톱, 탑재장치 인증 아님).
- 예측기 단독 학습: LightGBM, TCN, 기존 예측기에 잔차를 학습하는 Transformer. 잔차 모델 초기 확률이 v7와 오차3.7e-9로 일치. PPO 손실은 새 예측기에 전달하지 않음.

## 초기 engineering 결과 (최종 일반화 아님)

`artifacts/evaluation/*/4516f154ea039c09/selection/summary.json`
같은 9비행(8지역)에서:

| 모델 | 5초 AP | any3~7초 AP | 3초 선행 사건 R | 오경보/중계드론h |
|---|---:|---:|---:|---:|
| v7 64회 | .621209 | .286159 | .035698 | 4.66468 |
| 새 LightGBM | .635048 | .304902 | .068213 | 6.74300 |
| 새 TCN | .590515 | .280421 | .011596 | 2.12451 |

약한 TCN 결과도 보존. 대규모 개선 또는 실사용 달성을 주장하지 않음.

## 현재 실행

`python -m forecast_rebuild.campaign_train` 숨김 네이티브 프로세스. 실행 시작 PowerShell PID33372, 실제 Python PID는 status가 달라질 수 있음.
상태: `artifacts/campaign/status.json`.

- expanded_core: 학습535,200·선택216,000·보정180,000 표본, 특징646개.
- CPU: expanded_lgbm_base/energy 완료. 선택 샘플 AP는 base .6255/.2854, energy .6268/.2876. 최종 전체 비교는 아직.
- GPU: expanded_residual_base 완료, expanded_residual_energy 진행 중(각 최대5000 step).
- planned 특징 준비 병행(363기록, 약248개 완료 시점). 훈련은 고정 sampled rows만 전파 계산, 선택·보정은 전체 계산.
- 모두 끝나면 expanded_planned 표본을 만들고 planned LGBM·잔차 Transformer 두 모델 추가 학습.
- 총6개 후보. 자동 재시도·배포·최종 평가 개봉 없음. source hash guard가 있으므로 실행 중 features/dataset/models/train_* 파일을 수정하면 해당 학습이 실패한다. 현재 관련 수정은 마무리함.

## 남은 필수 작업

1. calibration의 freeze 파일에 대한 SHA 검사를 prepare_data/include-confirmation, evaluate confirmation, calibration.apply에서 실제로 강제하는 검증 함수 추가. 아직 existence 검사만인 곳 있음. Freeze 이후에는 해당 소스를 수정하지 않아야 함.
2. 경보 평가가 기존 v7 strict first-alert 규칙과 동일한지 무작위 예제 검사 추가. calibration.meets 및 금지/결측 API 검사.
3. 캠페인6모델 완료 기다리는 동안 독립 측정 스트레스(GPS bias2m, 센서잡음/결측) 구현 여부 판단. 측정 변경 시 공개지도 기하도 같은 믿음 좌표로 재계산해야 하며 깨끗한 기하를 그대로 쓰면 안 됨.
4. core/planned sample manifest의 records로 불변 evaluation_dataset.json 작성. 모든 모델과 baseline을 같은 선택24비행에서 전체 평가. 기존 model.json 단독 early stop 샘플 점수와 혼동 금지.
5. 선택 자료만으로 최종 모델 결정(앙상블 추가 시 가중치도 selection only). 모델 고정 후 baseline과 선택 모델의 보정20비행을 evaluate.
6. `calibration.fit`: 각 모델 보정 확률 및 경보 문턱. 센서 모델은 측정이 실제 유효한 링크만 경보 자격, 모든 실제 사건은 분모에 유지. 개발 목표 P90/R80/FA<=1 중계드론h/선행3초/최소50경보. 교수값·달성값 아님. 실패하면 enabled=false, 연구용 advisory 문턱만 기록.
7. `calibration.freeze` 후에만 확인16비행을 `prepare_data --include-confirmation`으로 읽고(선택 모델이 plan을 쓰면 prepare_plans --confirmation), 불변 확인 dataset 생성. 선택 모델·baseline 두 개만 확인. 확인 결과로 모델 재선택 금지. 최종32지도는 계속 봉인.
8. 확인에도 같은 고정 보정/문턱 적용. production_ready는 hardware 미검증이므로 false. 큰 개선이 없으면 솔직히 보고하되 실제 코드·학습·비교·안전한 실행 API까지 완료할 것.
9. 원본 해시/새 소스·모델 SHA/자료 분할/불변 비교·스트림 일치 검증. 한국어 최종 보고서 README/RESULTS 작성 및 사용 가능한 후보 정리. 아직 최종 답변 금지, 진행 중임.

네이티브 백그라운드 실행은 `Start-Process -WindowStyle Hidden`, 이 환경에서는 `exec_command(require_escalated)`로 시작해야 지속된다. 승인 검토는 허용됨. 사용자에게 같은 허락을 다시 묻지 않는다. 다른 채팅/메시지는 보내지 않는다. 하위 에이전트는 사용자가 명시 요청하지 않았으므로 생성하지 않는다.

## 02:00 추가 기록 - 우선 참고

- 원래 6후보 본학습 모두 완료(01:43:53). `campaign/status.json = training_completed`.
- 추가 7번째 `expanded_dense_residual_energy`도 5000 step 완료. `dense_truth.py`는 304,800개 훈련 표본의 미래 1~8초 min SINR 보조 정답을 생성했고, 1,850,693개 RF pair의 원래 이진 링크 정답과 완전 일치 확인. 미래 값은 손실 정답뿐, 입력에 없음. meta family `dense_residual`, 런타임 로더 지원 완료.
- `selection_benchmark.py`가 기존 혼합24비행에 baseline+6후보를 평가하는 중/완료일 수 있음. 이 결과는 엔지니어링용으로만 보존할 것. 향후 최종 cohort는 아래처럼 수정됨.
- **중요: 원래 collect.requests의 validation 순서(selection,calibration,confirmation)와 kind=(continuous,intermittent,clear)[i%3]가 맞물려 각 fold가 한 간섭 종류로 치우친 설계 문제를 발견. 확인 예측 결과는 아직 전혀 개봉하지 않았음.**
- `DATA_COHORT_CORRECTION.md`에 수정 원칙 기록. `collect_balanced.py` 숨김 새 프로세스 PowerShell PID29948. `balanced_collection_status.json`, stdout/stderr 참고. 각 검증 지도에 빠진 두 유형을 추가하는 **88비행** 실행 중(02:00 약4/88). 이미 수집한139개는 유지. 세 유형은 같은지도/방향/seed, 정책은 동결v7 64.
- 최종 **새 비행만** 사용: 선택15지도×3=45비행, 보정15×3=45, 확인14×3=42. 과거 historical validation16비행은 엔지니어링용으로만 남김. 기존32최종지도 계속 봉인.
- Inventory는 collection_manifest + balanced_collection_manifest 모두 연결하도록 수정됨. Balance 완성 후 prepare_data --workers3(확인제외), prepare_plans --workers2(기존cache reuse) 재실행해야 새 선택/보정60비행의 mmap/plan이 준비됨. 훈련319비행과535,200 표본은 그대로, 재학습 필수 아님.
- calibration.audit_cohort가 fresh_v7 source, fold 전체지도, 각map에 clear/continuous/intermittent 정확히1회씩인지 검사. 따라서 기존 혼합24/20 결과로 calibration.fit/freeze를 하면 거부됨. 새 **균형 cohort** 평가 dataset을 별도 작성해야 함.
- 균형 dataset: `dataset.json` 중 role=train은 모두 유지, dev는 source_kind=fresh_v7만 채택. 확인은 아직prepared rows에 없어야 함. 선택45/보정45. 모델/경보 freeze 후에는 fresh 확인42만 평가. historical확인2개는 제외(prepare자체는16→44중 포함될 수 있어도 primary snapshot에서 제외).
- **새 전체평가 output paths**: 각모델, 새dataset의 key+sourceSHA 조합identity가 달라지므로 기존eval캐시와 혼합되지 않음. 원래15fresh 비행의 cached predictions를 새folder에 SHA/key 확인 후 복사하면 재예측 비용 절약 가능.
- baseline과7후보를 balanced selection45에서 비교. 이후 최고tree와최고residual(또는dense) 간 weight .25/.5/.75 앙상블을 selection에서만 비교 가능. `blend.py` 구현 완료. 모델 choice는 선택 자료만. 보정/확인으로 재선택 금지.
- calibration.verify_freeze 추가됨. source/model/alarm/cohort policy SHA 검사. prepare_data --include-confirmation, evaluate confirmation, prepare_plans --confirmation 및 calibration.apply에서 강제. `confirmation_access.json` 별도 marker도 추가됨. **Freeze 이후 source에 적힌 파일 수정 금지**.
- 센서후보의 calibrated alarm support는 RF지원+own/peer 에너지실제유효+age<=2s. 실제사건은 분모에서 빼지 않음. Runtime.qualified_first_alert도 같은 gating. Advisory와 trusted alarm 분리.
- 런타임 `LocalForecast`는 sourcehardware 미검증 기본 false. 측정없음은None, 의도숫자범위검사, 센서결측값정리, 모델주파수/고도 profile 검사. 추가 tests10개 통과. 실제배치/스트림 및 plan 동등성 통과.
- 아직 남음: balanced dataset prep→fullselection+blend→winner/baseline calibration45→freeze→confirmation42→결과/개선기여/실사용미달여부 정직보고. 어떤 경우든 hardware 미정이므로 production_ready=false. 최종답변 아직하지말것.

## 02:50 추가 기록 - 아래가 최신

- 균형 보완 88비행 완료. 기본139와 합계 새227비행. `balanced_benchmark` 완료. 최종 primary 개발 selection45/calibration45; 훈련319비행535,200표본 유지.
- 선택 전체45비행: old AP5 .60672 /AP37 .25952, LightGBM base .6403/.2964, energy .6407/.2985, planned-energy **.692918/.370840**. Planned residual .6619/.3453. Dense 및 .25/.5/.75 혼합도 이를 넘지 못함. 최종 선택 **expanded_lgbm_planned_energy**. 선택 결과 `artifacts/completion/selection_decision.json`.
- `finish_study.py` 숨김 프로세스 PowerShell PID33740. 후보선택→보정45→freeze까지 완료. 02:49 기준 confirmation plan 32/44 준비. `artifacts/completion_status.json`은 prepare중 `freezing`으로 남아 있지만 정상. `artifacts/finish_study.stdout/stderr.log`와 `completion/prepare_confirmation_plans.*` 참고. 확인 primary는 새42비행만이며 historical2 제외.
- **이미 model_alarm_freeze.json과 confirmation_access.json 생성됨. 동결된 core 파일(features/targets/models/inference/dataset/metrics/calibration/runtime/planning/online_planning/evaluate/prepare_data/prepare_plans/sensors/contracts/beacon_codec/__init__) 절대 수정 금지.** 확인 결과에 맞춘 선택·보정 재조정 금지. 최종32지도 봉인.
- 보정 결과: new advisory 문턱 .79, 경보P .8308/R3 .02543/FA .76753/h; fullgoals미달enabledfalse. 고정밀 보조문턱 .81 P .90066/R3 .01591/FA .25872/h. Baseline advisory도 만족못해 .49fallback P .5305/R3 .18816/FA25.08/h. 이것은 calibration 결과, 최종확인결과와혼동금지.
- `calibration.fit`은 main fullgoals→advisory(.8P/5FA)→.5fallback 기존규칙 유지. 별도 high_precision_partial_point(90P/1FA/min50, recall조건제외)를 추가했고 **확인 전** 정책에 고정. `apply`가 main+partial 둘다 보고. sourcecase별카운트와Brier/ECE도 추가됨.
- 새 `diagnose.py`: 관측자격상한·단절지속시간별감지·지속highriskcoverage진단. latter는최초경보R대체아님. 선택관측상한legacy90.49%/energy90.30%, 달성성능아님.
- 15 unit tests passed (`test_core`, `test_codec`, `test_diagnostics`). Old strict first-alert counts와동일확인.
- 최초 selected runtime 검사에서 **확률차이3.448e-5** 발견. 원인은 public-plan feature 한 float32 ULP(1.19e-7); offline은x/c float64, online일부float32. frozen core를 바꾸지 않고 **runtime_stable.py** 어댑터가같은float64변환적용. `StableLocalForecast`, `load_research_candidate`. 전체한비행3362프레임·10355지원링크예측에서 **특징0/확률0/경보결정0오차**. `selected_runtime_checks.json`, CPUcold247ms warmmedian4.015ms. missingRFNone/missingenergyqualifiedfalse/trustedfalse통과. 추가wrapper SHA는최종audit와research_candidate에별도기록, 동결평가결과변경없음.
- `post_checks.py` 최초runtime실패후위수정으로재실행 PowerShellPID20268. 이미통과한runtime검사재사용, 지금 selection6비행측정stress중. 에너지없음/추가잡음2.5dB/고정GNSS축별sigma2m. GPS시모든기하·거리·명목경로특징 noisybelief에서재계산. 물리적비행·정답은같음. 로그 `artifacts/completion/measurement_stress.*`, `post_checks_status.json`.
- post_checks는stress→completed_comparison대기→analyze_results→audit 자동수행. **아직RESULTS보고서미생성/최종답변금지**. 오류나면해당로그확인. `analyze_results`는확인42의old/newgroup성적, 공간그룹(14map=12group) bootstrap10000, 관측자격상한/심각도진단작성. `audit`는원본legacy해시+451primaryrawSHA,8cache/label전체대조, selectedselection+cal+confirm/baselinecal+confirm cached각12rows새추론재현확인.
- `report.py` 구현. post_checks모두끝난후 **python -B -m forecast_rebuild.report** 실행해야 `RESULTS_20261102.md`, `artifacts/research_candidate.json` 생성. 보고서표행연속개행처리수정됨. 생성후숫자·해석검토필수. `QUICK_START.md`, `METHODS.md` 작성됨. QUICK_START는현재StableLocalForecast직접로드예시, 편의load_research_candidate도있음.
- 이전매뉴얼의intermittent를정확히정정: 15~45%start/65~90%end **한번의송신구간**, 버스트반복아님. ground2m −3..3dBm, continuous allmission. deviceactualunknown.
- Remaining: finish confirmation42, stress6, analysis/audit, reportgenerate/verify, bundleloadtest, README최신화, finalstatuscompleted+powerguardrelease. `research_status.json state=completed`는진짜끝났을때만써서 PID27024 sleepguard해제확인. trainingprocesses이미종료. 비교fixedoldpolicy only, 새연결율성능주장금지. FinalKorean실제AP/경보한계/원본보존/reportlink간결히. 실사용목표미달시명확히말하고대규모실용성주장금지.

## 03:00 완료 기록

- 독립 개발 확인42비행 완료. 공통5초AP **.5627853→.6520984 (+8.93%p)**, any3to7AP **.2352434→.3312922 (+9.60%p)**. 같은정답·2.43M/2.45M관측.
- 원래.5최초경보: P54.46→52.80%, R3 5.76→20.56%, FA/h7.21→27.37. 높은탐지와증가한오경보를함께보고.
- 고정cal 문턱 .79 new: P83.4515%, R3 **2.3053%**, FA/h.677474, 423경보. goal미달. 고정밀부분점.81도확인P86.36/R1.25로90%미달. Baselinecal fallback.49는P50.48/R15.24/FA22.31. 각자운영점차이명시, 같은FA예산우위주장금지.
- `analyze_results.py` 최초zero-positive flight AP=None처리누락으로실패. None을0으로바꾸지않고정의가능비행수명시하며평균/재표집에서제외하도록수정. 회귀검사추가하여**16tests pass**. 평가/동결core변경없음.
- 공간그룹12짝지은bootstrap10000: macrogroupAP5차+9.23%p,CI[7.79,10.97], AP37+10.54%p,CI[8.30,12.54]. pooledAP차이의CI로혼동금지. 기존7지도/추가7지도에서모두개선.
- selection6비행스트레스완료. macroAP5clean68.66,energy없음68.70,추가잡음68.67,GNSS축별2m65.76%. sensor없음qualifiedalarm0은센서자격차단결과이며탐지성공주장아님.
- `audit.json` **passed**,451원시archive/187기존의존파일SHA통과, source보존21+23, 8원시cache정답완전대조,selected1584/baseline1044개fresh추론cache재현. selecteddelta0,baselinemax3.576e-7. 공간train/dev그룹중복없음. wrapper/testSHA추가.
- Stable loader의실제bundle해시검사·모델로딩·20초첫추론통과. runtimefullmission측정은이전3362frames10355supportedlink,feature/probability/alarm0오차.
- `RESULTS_20261102.md`, `METHODS.md`, `QUICK_START.md`,최신README, 구조화final_results/research_candidate,비교PNG/SVG와신뢰도PNG완성. plot시각확인·보고서숫자확인완료. PDF는이번요청새로없어서추가생성하지않음.
- 추가perfect-future-window진단selection45: 정답그대로넣은최초경보R85.145% (expanded89.84%),P100. 실제예측성능이아닌내부reachability sanitycheck만수행,주요보고서에달성값으로포함하지않음.
- 장시간학습/비교모두종료. 마지막작업은완료상태기록과powerguard해제확인. 실사용목표는미달로명시하고연구구현·학습·검증의완료만보고할것.
