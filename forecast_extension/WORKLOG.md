# v9 연장 학습·새 경험 추가 작업 기록

최신 기록: 2026-11-02 11:15 KST. 사용자 요청은 이전 제안의 세 단계(Transformer 연장→새 비행·간섭·이동·위치 오차 추가→정밀도·선행재현율·오경보 비교)를 순서대로 **실행**하는 것. 사용자 재승인 질문 없이 완료까지 진행한다. 하위 에이전트는 명시 요청이 없어 사용하지 않는다.

## 보존·범위

- 작업 폴더 `forecast_extension`(v9)만 새로 작성. 기존 `forecast_rebuild` v8와 v5/v6/v7의 코드·가중치·보고서 보존. parent freeze를 `verify_protocol`이 확인한다.
- 최종 평가 지도 32개 계속 봉인. 새 확인은 기존 확인 지도14개에서 **새 난수·반대 진행 방향**의42비행. 새 지형 성적이라고 표현하지 않는다. v8 확인42의 결과는 학습·선택에 쓰지 않는다.
- 새 예측기만 지도학습. PPO는 v7 best64로 고정하여 모든 예측기를 같은 비행에서 비교. 실사용 인증·새 폐루프 연결률 성적 주장 금지. 하드웨어 미선정, production_ready=false.
- 현재 원시 기준 v8 독립확인 AP5 .562785→.652098, AP37 .235243→.331292. 고정cal 경보P83.45/R3 2.31/FA.677h로 목표미달. 이 숫자와 v9 선택용 점수를 섞지 않는다.

## 1단계 완료

- 기존 `expanded_residual_planned_energy/latest.pt` step5000을 읽기 전용으로 사용. optimizer·torch RNG 복원. 새 학습률 계획: 250update warmup, backbone3e-6/timing1e-5/head1e-4 peak→cosine. 총20000까지 고정15000추가updates, batch256.
- 새 balanced probe: 선택15지도×3조건=45비행에서각900표본, 40,500개. 현재 비교의공통legacy support로추출, 5초는currently_up제한. 학습probe12000개도고정. 원래535200훈련표본은복사·수정없이사용.
- 초기사전점검은training probe 무작위 읽기 순서로Windows mmap캐시thrash발견. **동일 표본을정렬**해속도수정. 처음 preflight_two_updates 프로세스만종료. 새 preflight_fast_two_updates 실제GPU2update·검증·저장통과.
- 본학습 native PowerShell PID34400, 로그 `artifacts/same_data.stdout/stderr.log`, 모델 `artifacts/models/same_data_20000`.
- **11:09 본학습20000완료**, best15000. 같은probe:
  - 5000: AP5 .6624594, AP37 .3487425, mean .50560095
  - 15000: AP5 .67628645, AP37 .35691571, mean .51660108
  - 20000: AP5 .67197379, AP37 .35666442, mean .51431911
  - 최고평균+1.10%p, 5초+1.38%p/3~7초+.82%p. 전체45비행평가는campaign이현재별도로수행중.
- **train.py/samples.py/contracts.py는 본학습·새자료학습 소스핀 대상. 새자료 C 실행 중 수정하면 실패한다. 이미정리되었으며변경하지않을것.**

## 현재 실행: 2단계

- 임시절전방지 `forecast_extension.power_guard`, PID12324. `artifacts/status.json state completed/failed/stopped`면해제, 최대12h. 디스플레이절전허용. 마지막에실제로끝나면completed기록해해제확인.
- **campaign native PID1372**, `python -u -B -m forecast_extension.campaign`, 로그 `artifacts/campaign.stdout/stderr.log`. usercomputerhidden창.
- 11:09:56 stage2 시작. **11:15 collection31/377**, streamingprep24/335 정상, 오류로그없음.
- 수집377: 훈련245지도 각새비행1개 +선택45+보정45+확인42. train은4대형분산/3간섭유형분산, dev는각지도3간섭유형모두. 새seed, dev방향v8반대. 모든비행oldv7fixedpolicy. 원래환경2.4GHz/50m/강도−3~3dBm/고정지상간섭이며intermittent는한송신구간(버스트아님).
- `collect.py` native child4CPUworkers, 원래검증된`adaptive_joint_expanded.experiment._worker` 읽기전용재사용. 메타/아카이브해시확인후신규manifest. 새로운raw는ownartifacts/raw/fresh. 확인raw는수집만하고모델평가·학습cache는freeze전금지.
- `stream_prepare.py` native child3workers. 완료된각비행을즉시compile/plan/aug하여수집과병행, 확인role제외. 335nonconfirm만준비. 엄격히새source메타&sensorreplay24frames확인. pipeline끝`data.prepare`는다시검사후cache재사용.
- 훈련84비행에추가관측variant. 63GPS(축별sigma1/2/4m각21), 21sensor-only. 센서clean/noise추가2.5dB/missing. GPS센서유형교차균형. 84each clean1200+aug1200 **서로다른** time/linkrows, 나머지161flight2400. 새표본예상588000, old535200과총1123200.
- 위치오차시관측좌표/상대거리/반경속도/현재GIS손실/46goalgeometry/82publicplanfeatures모두같은noisybelief에서재계산. 자기명령offsetbias반영, **수신되지않은이웃명령은0/결측유지**(이전stress보다개선). selfintent변경없음. 미래위치·간섭원정보는입력없음. 라벨·실제물리는고정.
- 실제기록3query에서GPS2m+energy미싱전처리 사전검사통과(`preflight/belief_check.json`), labels완전동일/features3×728/seq3×40×111, unknownintent유지.
- `test_belief`2개, `test_metrics`1개(30무작위상황) 통과. **metrics.py**는old최초경보규칙을maxaccumulate/searchsorted로벡터화. NaN/결측/히스테리시스/경계포함old와모든counts완전일치확인. 정의변경없음.

## 자동 pipeline 구현 (campaign.py)

1. B20k완료기다림(이미완료)→collect&stream 시작.
2. 동시에GPU로기존neural4750/Bbest15k/v8tree를oldselection45에서전체평가. `stage1_comparison.json`저장. 수집과독립계산, 새자료미학습모델.
3. collection377+stream335완료→`data`재확인→`build_samples`new_only→merged_trees.
4. C `new_data_20000`는**같은step5000weights/optimizer에서시작**, old+new자료로동일15000추가update·동일LR계획. **numpy batch sampler는새SEED로설정되어B와다름**, 이를순수인과효과라고단정하지않음(비교계획에명시). GPU C와CPUnewtree energy/noenergy두모델학습병행.
5. `selection.select`: old+fresh선택90비행. 기본7후보 baselinev7/v8tree/v8neural/B/C/newtree/newtree_noenergy. 최고APtree+최고APneural의 .25/.5/.75 혼합3개추가, 총10후보.
6. **선택기준** 비교계획 `artifacts/comparison_plan.json`에수집전고정. allgoals가능하면R최대, 아니면P>=90/FA<=1h/min50경보조건에서R최대, 없으면AP평균. AP최고모델별도기록. rawselection label만사용, cal/confirm미사용.
7. 신선한보정45에서모든10모델 Platt/thresholdfit. raw.5, 기존advisory, **동일FA예산1/h**, P90FA1partial모두보고. 5초cal은currentlyup으로범위일치, 37은allstates. 전체사건분모유지.
8. **freeze.json**에모든fixedmodel/checkpoint/threshold/selection/cal/sourceSHA한번에고정→새confirm42cache/plan→모든10고정후보를한번확인. 확인결과로재선택금지. `completed_comparison.json`작성.
9. campaign은끝에status **running/comparison_complete_pending_audit**로남김. **아직audit/report/runtime검사모듈작성필요. 최종답변은모든작업실제완료후.**

## 새 코드·주의

- evaluate.py: parentPredictor를재사용하나 ownmodel/input/cache/sourceidentity로캐시. rawbaselinev7는같은weightsSHA의원래수집aux_probs사용가능. 최종감사때fresh재추론일부대조해야함.
- ensemblemodelpin은재귀membermanifest/weightsSHA검사. ensemble평가시같은datasetSHA의원래memberprediction캐시를읽어혼합해중복GPU계산줄임.
- new_only samples는fresh329trainrecords(245+84) +newselection45. C CombinedSamples는oldtrain+newtrain(원래trainmatrix복사안함). source_flight_key로clean/aug같은비행을묶어중복과표집방지.
- merged_trees에oldmanifest의unuseddevrecords도목록에는있지만train_index의실제모든record.roletrain을검사함. train데이터는누출없음. meta목록과실제최적화행을구분.
- augmentation코드는부동소수메모리값을float16으로저장후같은표현으로계획특징생성. 미래없음. 하드웨어오차가아닌simulation가정.
- script수가많아버그가능, 로그보고고칠것. campaign이evaluate/calibration/selection을시작시import하므로 **그파일을수정하면현재프로세스에는반영되지않음**. 해당핵심버그수정시freeze전campaign재시작필요. collect/stream이진행중이면중복실행/원본cache동시쓰기주의. 현상확인후ownprocess만제어, 사용자에게중복승인요청하지말것.
- 기존freeze32지도를열지않고새이미알려진14지도에서freshseed/dir평가. 새로운지형일반화/실사용보장을주장하지않음.
- 새`01_예측연장_현황.bat`가v9상태. 기존00bat는v8완료상태. 사용자에게구분필요.
- native Start-Process는이환경에서`exec_command(require_escalated)`, `-WindowStyle Hidden`로실행해야유지됨. Pythonchild는CREATE_NO_WINDOW. 가상환경 `.\drone_forecast_joint_training\.venv\Scripts\python.exe` 사용. WindowsUnicodeLightGBM모델파일은PythonUTF8쓰기/읽기.

## 남은 작업

- 진행중377수집·stream·C학습·10후보cal/confirm정상완료까지지속. 오류는원인고쳐재개, 미완성에서최종답변하지말것.
- 최종audit.py/report.py/figures.py/런타임bundleloader(필요) 작성. 비교는훈련연장효과와새경험효과, 같은FA예산에서P/R/FA, rawAP/thresholdtradeoff, 새지도아닌새비행scope명시. 표본APNone양성없음은정의불가로처리.
- 최종선정모델은이미freeze에서정해진것유지. 성적보고모델재선택금지.
- 전체한비행batch/live검사: parent **runtime_stable.StableLocalForecast**를새model/policy직접생성하면동일728feature+float64plan지원. parentload_research_candidate는v8bundle을읽으므로v9용직접loader새로만들것. hardwarefalse/trusteddisabled 유지.
- 원본187SHA+F8freeze/legacy기존모델보존, 새원시377SHA, 학습/선택/보정/확인그룹, sampleindices trainroles, augmentationlabels동일, predictioncache재추론검사.
- 이턴사용자요청PDF없음. 상세MD+standalonefig로충분. 원하면나중PDF. 사용자최종에는실제개선·미달목표·보고서간결히.
- 완료후ownstatusstatecompleted로powerguard해제확인, 실험백그라운드프로세스없음확인. 실사용성능불충족이면분명히보고하고단순시간투자로90%보장주장금지.

## 12:05 최신 기록 - 이 부분 우선

- 1단계 전체oldselection45평가완료. A oldneural AP5 .6618560 /AP37 .3453269 /R3 .246594 /FA32.695h. B longbest15k AP5 **.6725265** /AP37 **.3560340** /R3 .245883 /FA29.641h. 기존트리 .6929183/.3708403/R3.246594/FA31.137h. 모두**선택용자료**, 독립확인아님. 연장만으로+1.07%p정도, 기존트리못넘음.
- 새수집 **12:03 343/377**, stream335nonconfirm준비완료(아직수집종료대기중). 12:01보정까지완료거의335, 이제confirmraw만수집. 확인예측결과미개봉.
- **수치 경계 수정**: fastmetric np.float32 score와Python float64 quantilethreshold 비교에NumPyweakscalar가float32로문턱을반올림하는문제발견. old np.where는float64비교. `.metrics`를score명시적float64변환하도록수정. .5/.25고정문턱은정확표현이므로기존stage1결과변경없음. testfloat32+nextafter 경계추가, 총**4개신규tests(30random+경계+관측2)**통과. 기존정의그대로, 새기준정의변경없음.
- 이미실행중maincampaign은이전fast함수를import한상태였으므로**제어기만정상handoff**. Oldmainproxy1372/actual39428만종료. 수집·stream은계속살아있음. **현재 제어기PID15992 `forecast_extension.resume`**, 로그 `artifacts/resume.stdout/stderr.log`.
- collect proxy26332/actual12512, stream proxy20028/actual29584(정확매칭CIM확인필요할수있음). 모두기존raw작업연속. **campaign을다시실행하지말것**(중복작업위험).
- `resume.py`: collection_status+stream_status완료대기→data재확인→new_only/mergedmatrices→Ctrainingstart+tree2개→C완료→**별도 fresh `finish.py`**로selection/cal/freeze/confirm. 따라서최종평가는수정된float64 fastmetric을신규프로세스에서읽음. `finish.py`가completed_comparison쓰고post가이어짐. marker`CONTROLLER_HANDOFF_REQUIRED.json statecompleted`.
- **post nativePID28824** `forecast_extension.post`, 로그 `post.stdout/stderr.log`, `post_status.json`. 완료비교대기, dataset.json생기면C훈련중에`reuse.py`로oldA/B/tree의동일old45예측캐시를각각model/input/sourcehash확인후새90selectionfolder에복사(중복GPU계산절약). 이후runtime→audit→analysis→stress→figures→report→statuscompleted, powerguard해제자동.
- Powerguard proxy12324여전히유지, 최종statuscompleted때해제. runtime연구용 false,실사용미검증.
- **아래post모듈구현완료**: runtime.py(부모StableLocalForecast계승+v9load), check_runtime.py(새선택비행전체batch/live), audit.py(741예상원시hash+84증강+187orig+trainroles+모든10캐시재추론), analyze.py(새비행조건별·paired12spatialgroups bootstrap), stress.py(선택6비행fixedqueries<=6000의B/C/winner ranking오차진단), figures.py, report.py, README. 최종자료아직없으므로실제성공여부확인·버그수정필요. 최종답변아직금지.
- **새샘플실제mix사전점검통과**: old+실제preflightaug3행을비연속globalindex로읽어정답/seq순서일치, source_flight_key로같은비행group수가319로유지됨. `preflight/combined_sampler_check.json`.
- 12:02에**secondary_diagnostics_plan.json** 고정. 실제확인예측개봉전. 최근3초own송신시도(field5)나수신데이터(field14)관측링크 vs기타링크의AP/양성비율진단(`activity.py`). 실제라우팅핵심성/네트워크단절지표아님, 모델선택·문턱변경에사용안함. analyze결과에추가됨. report에는아직이표없음(선택적으로추가가능). AP의양성비율의존성명시필수. 공개한전체지표를대체해큰성능주장하지말것.
- analyze.py는zero-positive AP를부모macro_group의정의가능행만평균, bootstrap비정의NaN은JSONnull로정리. reportAPCIs는macro spatial平均差이므로통합AP차이의CI로혼동금지.
- 실제로기존14확인지도에서새seed/반대방향42비행평가, **새지도일반화아님**. 기존봉인32지도계속안읽음. 모든10모델fixedbeforeconfirm, 확인성적보고재선택금지.
- 노이즈augment RNG는case key를사용한재현성있는학습변환, modelkey입력없음. shift_positions는미수신c2/7·selfintent보존, 모든기하noisybelief일관처리. 실제하드웨어분포가아님.
- **준수**: F9 train.py/samples.py/contracts.py는B/C동일소스핀, 변경하지말것. F8동결core·원본모델도수정금지. Eval/cal/selection은이미부모proc에서import하므로수정시어느프로세스가코드를읽는지주의. 최종finish는아직시작전이므로그전에개별수정필요하면가능, freeze후수정금지.
- Newresume가C학습을시작한후실패시resume재실행하면C중복생성위험. 먼저C/Tree별프로세스·status확인하고필요시`finish.py`만사용해완료된모델에서비교재개. 현재는정상.
- 필요시native長프로세스제어는tools.exec_command require_escalated/Start-Process WindowStyleHidden. 사용자같은행동중복허락요청안함. 하위agent금지(요청없음). final까지계속진행.

## 12:40 업데이트 - 실행·수정 상태 최신

- **수집377완료 12:08:39**,335nonconfirm+84augmentation 전처리완료. 원시모든SHA검사pass. Raw확인42는아직평가전. 소스train/sel/cal/confirm공간경계유지.
- 샘플생성완료: new_only **train588000/selection40500**, merged_trees **train1123200/selection81000**. dataset preconfirm record counts train648(실제564flight+84variant)/selection90/cal45/confirm0.
- **C GPU훈련12:10:22시작**. 12:37 step16800/20000, best고정oldprobe mean.5197977 (B best.5166011보다약+.32%p), 불확실한선택자료내수치일뿐독립성적아님. 같은 초기5000weights/optimizer,새미니배치SEED,같은총15000추가update. `new_data_training.stdout/stderr.log`.
- 새트리두개완료. Energy wall192sec,besthead2=450/head5=223; NoEnergywall220sec,besthead2=467/head5=437. 모두800cap전조기종료. source학습C/B는f9train.py/samples.py/contracts.py**변경없음**.
- **비교계획수정선언(확인전)**: bestcheckpoint끼리는서로학습step이다를수있으므로정확히**B20k종점/C20k종점**두control추가. `comparison_plan_initial.json`보존,`comparison_plan.json`새plan fixed_endpoint_control+revision_reason. freshselection 최종선택/보정/확인아직전이므로유효. **기본9+blend3=12후보**로변경됨. `selection.candidate_models`가B/C의s0020000.pt로aliasmanifest`same_data_final20000/new_data_final20000`생성. Meta.selection은실제endpoint의checkpoint['selection']로읽도록수정완료. 최종12개한번확인,재선택금지.
- **평가CPU중복계산최적화**: 원래한비행54,460queries의staticfeature추출1.60초실측. 12모델반복을줄이기위해`feature_cache.py CachedEpisode`추가. F8원래feature함수를그대로한번계산해float32저장, Querylookup으로읽음. energy없는후보는정확히energy24columnsNaN/flags0로초기화. planningfalse는646앞부분. 원래모델입력/정의변경없음. 디스크여유20GiB미만이면캐시포기하고기존함수로fallback. Cachelazy라baselineaux재생성엔불필요계산안함.
- 실제새선택비행128queries×energytrue/false×plantrue/false에서**전체featurebitwise같음**,oldtree확률도exact같음. `preflight/static_cache_check.json`.
- **evaluate.py현재새Cache버전**: `Episode=CachedEpisode`, cacheidentity implementation=dict(evaluateSHA,feature_cacheSHA,metricsSHA)로강화. Freeze.sources에feature_cache.py추가. 현재실행중resume는Eval을직접쓰지않고**freshfinish child**를나중실행하므로최종평가에새코드반영됨. C훈련F8Episode는영향없음.
- 기존Eval구현`source_snapshots/evaluate_before_static_cache.py`보존. `reuse.py`가등록된이전구현snapshot identity+원래input/modelSHA검사하여**예측배열만**동일새inputcohort폴더에복사. .5통계는어차피새코드로재계산. **수정후새identitytarget에도135파일 재사용완료**(12:30). 이전prefillfolder는부작용없이보존. cachepred를다른난수/다른가중치에재사용하지않음.
- Post프로세스28824는11:39부터계속대기, `post_status.json`waiting. globalpowerguard12324유지. Resume15992가C끝나면**finish.py새프로세스**로고정선택/보정/확인실행. Eval/Selection/Calibration은이freshchild에서시작하므로이전mainimport스냅샷과무관. **새코드재수정필요하면finish child시작전이안전**,freeze후핵심수정금지.
- finalaudit.py보강: old/v9원시캐시8기록 values/masks/intents/links완전대조+labels재계산, originalmapPLANsha와부모187source/weights·testmarker확인,84aug truth/masks/c2unknown보존, 실제train_index의role검사, 모든12모델cached예측재추론. B/Csourcepins동일/초기reference동일/각20000step도확인.
- secondaryactivity진단은last3srawlocal5/14송수신관측여부, model선택/threshold에안쓰는사전선언분석. report에선oldbest/winner만표, 양성비율과AP동시표시. 실제routingcriticality/네트워크단절아님.
- 모든V9Python ASTparse통과. 신규4unit(pass): 관측일관2 +30random strict counts +float32/f64경계. ActualGPU2step保存+realepisodeaug+mixedold/newindices+losslessstaticcache 사전검사통과. 아직최종audits/runtime은미실행.
- **남음** C20k완료→freshfinish로12후보선택90eval/sweep→cal45각fit→freeze→freshconfirm42×12→postruntime/audit/analysis/stress/figures/report actual정상완료. 최종문서·그림숫자/레이아웃검토, bundleload test, powerguard해제/잔여process없음확인. 성공하기전최종답변금지. user하드웨어미정이라productionfalse유지. 현재전체상태정상.

## 12:57 비교 단계 병렬 입력 준비

- C20k 정상 완료12:44:01, 선택best15k. freshfinish가stage3선택90비행평가 진행중. 원본 유지, confirmation미개봉.
- 고정20k종점두모델(aliases) 예측만 `prefill.py` worker가병렬계산(PID27124/8632, actualchild별도). **exact768querybatch, model/input/implementation identity동일**, 임시파일에서원자적으로npz게시, summary/통계/선택안함. Mainfinish만평가통계/선택수행. 별도모델파일은읽기전용. 신규helper는학습/평가정의를바꾸지않음. 모든후보캐시는마지막auditfreshprediction대조.
- calibration 입력staticcache는 `prepare_static.py` PID39224가한worker로사전생성. probability/정답통계/문턱없음. 선택코드는calibration예측을안읽음. static producer경합금지, 본calibration시작전에완료확인.
- post는기존그대로최종고정비교완료대기중. status.py 표시만개선(완료후마지막running스냅샷으로오해하지않도록poststage/selectedstep출력).

## 13:27 최종 확인 진행 - 현재 우선 기록

- 선택90비행의12후보 raw/sweep 완료13:09:24. **선정 v9_blend_75 = new_data_tree75% + new_data_final20000(정확20k)25%**. AP최고는new_data_tree(.54468493mean), 선택ruleprecision90_FA1_partial_then_event_recall. 선택기준 P91.39%,R3 1.2978%,FA0.1968/h였으므로 전체목표는이미미달. 이는선택자료수치, 최종확인과혼동금지.
- 모든12freshcal45완료, **freeze.json 13:15:58** SHA8f8f7f1dc49f5f940b2d35104b552313696fc45c965cc9da10ef4713dd3a72e8. 이제freeze.sources에명시된core모듈(train/data/evaluate/metrics/selection등) **수정금지**.
- 확인RF+publicplans42준비완료13:22:50. 현재 **fresh확인42×12모델평가진행**. 13:26 oldneural28/42. Mainresume15992→finishchild는정상, logs/final_comparison.stderr빈파일.
- 속도향상helperprefill.py v1/v2완료, 원본helper snapshots보존. v3는confirmation을허용하되verify_freeze+allowlistedmodel+fixeddataset guard. 마지막직렬모델들의예측만동일768batch로계산, npz원자게시; 통계·모델·문턱선택은Mainfinish만수행. **확인prefill_A proxy32352**(Bbest→Cbest), **prefill_B proxy20544**(Bexact20k→Cexact20k). 로그prefill_confirmation_A/B.stderr, 각모델*_confirmation_status.json. 13:26첫모델26/42,오류없음. staticcache는oldtree가앞서모든42생성. Metadata/stat정의변경없고최종audit모든12cachefresh재추론.
- **전체비행 runtime 선행검사통과13:21:30**. nativeproxy29940이미종료. **3164frames,9834supportedlinkpreds, maxprob0/maxfeat0/alarmdiff0**, RFmissingNone/trusteddisabled. CPUcold763.4ms/warmmedian6.596ms. runtime_checks.json기록. check_runtime.py는freeze/checker/runtime/depSHA모두같으면post호출시해당완료검사재사용(반복검사안함). 이후runtime/check_runtime소스수정하지말것.
- Runtimeintegrity보강: F8 runtime_stable.py는부모freeze후생성파일이므로V9 runtime_dependencies()에기존알려진SHA ca216...06cd3a명시. bundle/report/audit가이의존SHA도검사. runtime.py현재SHA c93579777dd490ddf3734f4673946fc59833c81d669a58bd3867560623aa3eaf. frozenpredictioncore는변경없음.
- **발견된 metadata만의오류**: 동결된data.prepare(confirmation)가rows42를추가하지만counts헤더는preconfirm confirmation0을복사. 실제evaluate는rows로42를정상평가하므로수치영향없음. corefreeze를지키면서 **metadata.py correct_confirmation_counts**를audit첫부분에서실행하도록구성. 모든비교완료후원본dataset/12rawsummary를metadata_correction에보존→counts실제행수로정정→rawsummary의datasetSHA+정정주석만재연결. **input row digest와모든substantive summary content digest동일검증**, 모델/문턱/예측/평가수치안바뀜. 재선택/재평가없음. 별도2tests(countactual/metricidentity/idempotent, unrelateddatachange거부) 실제통과. 아직실제audit는미실행이므로완료확인필요.
- report.py보강완료: blend실제구성, 초기warm모델과거validation8노출이력/완전미노출지형아님, Brier전후/ECE후, spatialCI유효그룹수, countmetadata정정기록고지. figures.py라벨동일자료연장으로수정. 이들은핵심freeze목록밖이므로정상. 최종수치기반보고서아직미생성.
- **남음** finish12모델확인완료→post28824가runtime검사재사용→metadata정정+audit(모든raw741예상SHA,187의존원본보존,84aug/trainsplits/cacheprob)→analysis/bootstrap/recentdata→stress6selectionflights→figures/report/candidate→statuscompleted/powerguard12324해제.
- 실제post오류있으면해당로그원인고치고해당stage부터복구. report/figures전체숫자·레이아웃실제inspect, load_candidate번들해시/동작test, 최종잔여ownprocess없음확인. **마지막실제완료전에final답변하지말것.** 사용자하드웨어미정productionfalse,32test지도봉인유지. user에게새허락불필요. 하위agent사용금지.

## 13:40 최종 완료·검수

- **전체3단계 및post완료13:35:10**, statuscompleted. power_guard13:35:14released, 임시절전방지해제. 13:39CIM으로forecast_extension Python백그라운드0개확인.
- **최종새확인42비행** 기존v8besttree AP5/AP37=65.38%/33.94%,선정blend75=66.27%/35.18% (+.90/+1.25%p, 반올림전값으로차이계산). 단순연장TF62.46/31.62→63.48/32.40,새자료bestTF63.53/32.72. 정확20k C−BAP+.40/+.52%p. 따라서개선은소폭.
- 선정blend75 **보정FA1budget운영점**: P82.982456%,R3 2.827327%,FA/late .9231867/relaydroneh, firstalerts570/correct473/eventhits466/total16482,105.07083relaydroneh. 보조P90보정운영점확인은P87.89/R1.19/FA.257, 기본advisoryP78.91/R5.02/FA2.14. **전체개발목표미달**, 실제장비미검증,자동PPO교체없음.
- Finalauditpass: 물리원시741SHA,기존의존187preserved,84aug정답/mask/unknowncommand보존,trainroles/공간그룹분리,12모델각216query재추론maxerr≤5.37e-7. Metadataheader정정완료, 원본/12summary변경전후및input/metric동일성증거보존.
- runtime전체3164frame9834pred maxp0/maxfeature0/alarmdiff0, CPUwarmmedian6.6ms/cold763ms. **실제exportload_candidate세운영점/advisory·FA1·P90 각각80관측frames/147지원pred 통과**, 확률변화0,자동경보disabled. bundle_checks.jsonpassed.
- RESULTS_20261102.md전체수치대조/두PNG실제view_image검수완료. 최종frozen모델/threshold/운영정의재선택안함. 32test지도계속미개봉. 학습·확인종료후새추가학습안함.
- delivery_manifest.json에검수보고서/그림/protocol/freeze/audit/runtime/bundle/metadata수정기록SHA보관. manifest생성시figuresprovenance와final_resultsSHA, bundle검사와현재candidateSHA,최종freeze확인통과. README완료상태와bundle/manifest안내수정. 완료보고전추가작업없음.
