"""실제 결과 JSON에서 숫자를 읽는 항목형 한국어 HTML 결과 설명서."""
import base64,html,json,re,io,csv
from pathlib import Path
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json
from usable.evaluate import RESULT,RUNS

LABELS={'B0_legacy_inputs':'기존 입력','B1_observed_radio':'비콘 관측 보강','B2_multitask':'비콘 + 보조 과제',
        'B3_beacon_map_multitask':'비콘 + 지도 + 보조 과제','B4_beacon_map_single':'비콘 + 지도 선행 정보'}


def num(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def pct(x,n=1):return '—' if x is None else f'{100*x:.{n}f}%'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def card(value,label):return f'<div class="metric"><b>{value}</b><span>{label}</span></div>'
def block(title,body,id=None):return f'<section id="{id or ""}"><h2>{title}</h2>{body}</section>'
def bullets(items):return '<ul>'+''.join('<li>'+x+'</li>' for x in items)+'</ul>'
def image_tag(path):return '<img class="figure" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def figures(comparison,selected,dense,base,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    path=ROOT/'docs/assets/NotoSansKR.ttf'
    if path.exists():
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        static=ROOT/'cache/usable/plot_font_regular.ttf'
        if not static.exists():
            font=TTFont(path);instantiateVariableFont(font,{'wght':400},inplace=True);font.save(static)
        fm.fontManager.addfont(str(static));plt.rcParams['font.family']=fm.FontProperties(fname=str(static)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    result=[]
    keys=[k for k in comparison['results'] if k.endswith('_s11')]
    labels=[LABELS[k.rsplit('_s',1)[0]]+('\n(선정)' if k==selected else '') for k in keys]
    ap=[comparison['results'][k]['balanced']['all']['average_precision'] for k in keys]
    fig,ax=plt.subplots(figsize=(10.7,4.0));bars=ax.bar(labels,ap,color=['#1759ae' if k==selected else '#a1b7cd' for k in keys],width=.62)
    for b,v in zip(bars,ap):ax.text(b.get_x()+b.get_width()/2,v+.015,f'{v:.3f}',ha='center',weight='bold')
    ax.set_ylim(0,min(1,max(ap)*1.3+.02));ax.set_ylabel('AP (0–1)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('동일 평가 표본의 예측 성능 · 다섯 변형 모두 Transformer',loc='left',pad=18,weight='bold')
    fig.tight_layout();p=folder/'model_ap.png';fig.savefig(p,dpi=180);plt.close(fig);result.append(p)
    regions=sorted(dense['balanced']['by_region']);short=[x.replace('seoul_20261023_b2_','서울 B2-').replace('seoul_20261023_','서울 -').replace('bucheon_jungdong_east','부천 중동').replace('bucheon_simgokbon','부천 심곡본') for x in regions]
    old=[base['balanced']['by_region'][k]['average_precision'] for k in regions];new=[dense['balanced']['by_region'][k]['average_precision'] for k in regions]
    y=np.arange(len(regions));fig,ax=plt.subplots(figsize=(9.7,4.8));ax.barh(y+.18,old,.32,label='기존 입력',color='#aab8ca');ax.barh(y-.18,new,.32,label='선정 모델',color='#1759ae')
    ax.set_yticks(y,short);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('AP (0–1)');ax.legend(frameon=False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('전체 48개 평가 비행의 지역별 결과',loc='left',pad=15,weight='bold');fig.tight_layout();p=folder/'region_ap.png';fig.savefig(p,dpi=180);plt.close(fig);result.append(p)
    from sklearn.metrics import precision_recall_curve
    fig,ax=plt.subplots(figsize=(9.7,4.2))
    for name,label,color in [('B0_legacy_inputs_s11','기존 입력','#8c9db2'),(selected,'선정 모델','#1759ae')]:
        yy=[];pp=[]
        for f in sorted((RUNS/name/'dense').glob('*.npz')):
            with np.load(f) as z:yy.append(z['y']);pp.append(z['probability'])
        y0=np.concatenate(yy);p0=np.concatenate(pp);precision,recall,_=precision_recall_curve(y0,p0)
        # 선 개수 축소는 표시만 적용합니다. AP 계산은 전체 점수에 기반합니다.
        q=np.unique(np.linspace(0,len(precision)-1,min(3000,len(precision))).astype(int))
        ax.plot(recall[q],precision[q],label=label,color=color,lw=2)
    for key,label,marker in [('balanced','일반 경고','o'),('conservative','보수 경고','s')]:
        if dense[key] is not None:
            v=dense[key]['all'];ax.scatter([v['recall']],[v['precision']],s=65,marker=marker,label=label,zorder=3)
    policy=read_json(RUNS/selected/'alarm_policy.json')
    if policy['selected']:
        q=p0>=policy['selected']['enter_threshold'];tp=int(y0[q].sum())
        ax.scatter([tp/max(int(y0.sum()),1)],[tp/max(int(q.sum()),1)],s=80,marker='D',color='#b45336',label='고신뢰 알림 진입점',zorder=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='재현율 · 실제 위험 중 경고 비율',ylabel='정밀도 · 경고 중 실제 위험 비율')
    ax.set_title('경고 기준 변화에 따른 정밀도–재현율 균형',loc='left',pad=15,weight='bold');ax.legend(frameon=False);ax.grid(alpha=.15);fig.tight_layout();p=folder/'precision_recall.png';fig.savefig(p,dpi=180);plt.close(fig);result.append(p)
    return result


def build():
    selected=read_json(RESULT/'selection.json')['selected_variant']+'_s11'
    comp=read_json(RESULT/'bank_comparison.json');dense=read_json(RUNS/selected/'dense_evaluation.json');base=read_json(RUNS/'B0_legacy_inputs_s11/dense_evaluation.json')
    runtime=read_json(RESULT/'runtime_evaluation.json');cal=read_json(RUNS/selected/'calibration.json');run=read_json(RUNS/selected/'run.json')
    motion=read_json(RESULT/'motion_check.json');cost=read_json(RESULT/'beacon_cost.json');manifest=read_json(ROOT/'datasets/usable45/collection_manifest.json')
    confirmation=read_json(RESULT/'confirmation_results.json');alarm=read_json(RUNS/selected/'alarm_policy.json')
    existing_alarm=read_json(RESULT/'alarm_policy_existing_test.json');latency=read_json(RESULT/'runtime_verification.json')
    out=ROOT/'docs';out.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True)
    figs=figures(comp,selected,dense,base,assets);m=dense['balanced']['all'];old=base['balanced']['all'];tight=runtime['conservative_windows'];rte=runtime['conservative_events']
    body='<div class="hero"><div class="eyebrow">UAV COMMUNICATION · TRANSFORMER FORECAST</div><h1>통신 장애 예측<br>개선 결과와 사용 기준</h1><p>단일 Transformer 계열 · 지도 선행 정보 · 주기 관측 · 위험 점수와 알림의 분리</p><div class="metrics">'+card(num(m['average_precision']),'전체 평가 AP')+card(num(old['average_precision']),'같은 환경의 기존 입력 AP')+card(num(latency['one_observer_7_links']['median_ms'],2)+'ms','1대·7개 이웃 추론 중앙값 · 현재 CPU')+card(str(confirmation['flights'])+'회','정책 고정 후 새 난수 확인')+'</div></div>'
    body+='<nav>'+''.join(f'<a href="#{i}">{text}</a>' for i,text in [('outcome','결과'),('reason','개선 근거'),('data','실험 조건'),('operation','경고 운용'),('sim','시뮬레이터'),('use','사용 방법'),('limits','기타 설명')])+'</nav>'
    body+=block('1. 결과 요약',
        '<div class="callout"><b>적용 범위: 시뮬레이터의 연속 위험 점수와 보조 알림. 사전 자동 재배치의 단독 실행 기준으로는 부족.</b><p>AP 개선과 3초 이상 선행 탐지의 구분. 고신뢰 알림의 오경보 감소와 조기 탐지 감소의 상충. 평가 지역은 기존에 확인한 지역이며, 추가 확인도 같은 지역의 새 난수 비행.</p></div>'+
        table(['항목','기존 입력 Transformer','선정 Transformer'],[[
            '전체 평가 AP',num(old['average_precision']),num(m['average_precision'])],
            ['지역별 AP 단순 평균',num(base['balanced']['macro_region_ap']),num(dense['balanced']['macro_region_ap'])],
            ['일반 경고 정밀도',pct(old['precision']),pct(m['precision'])],['일반 경고 재현율',pct(old['recall']),pct(m['recall'])],
            ['일반 경고 FPR',pct(old['false_positive_rate']),pct(m['false_positive_rate'])],['Brier 점수 · 작을수록 유리',num(old['brier'],4),num(m['brier'],4)],
            ['전체 평가 창',f'{old["samples"]:,}개',f'{m["samples"]:,}개']])+image_tag(figs[0])+
        '<p class="caption">막대그래프: 지역당 최대 8,192개의 동일 무작위 평가 표본. 위 표: 48개 전체 비행의 모든 적격 창. 두 표본 집합의 구분.</p>'+image_tag(figs[1]),'outcome')
    body+=block('2. 개선 항목과 선정 근거',table(['문제','개선','근거와 확인'],[
        ['데이터 통신에 참여한 링크에 편중된 관측','드론별 1Hz 비콘과 역방향 측정 보고','관측 공백 감소 목적. 128B 비콘의 수신 성공 시점만 사용. 슬롯 비용 4% 반영.'],
        ['현재 신호만으로 곧 발생할 건물 차폐의 구분 한계','최신 이웃 위치·속도 + 정적 지도의 3초·5초 전파 손실 추정','미래 실제 위치 대신 등속 외삽. 비행 금지 구역 교차 표시를 함께 입력.'],
        ['지역 절대 위치에 대한 과적합 가능성','절대 좌표 직접 입력 제외, 물리 단위의 상대·전파 정보 사용','자기 좌표는 지도 파생 계산에 활용. 신경망 직접 입력과 구분.'],
        ['단일 이진 정답의 제한','5초 장애 + 10초 장애 + 미래 최소 SINR의 공유 표현 학습','B2 대 B1, B3 대 B4의 보조 과제 비교. B3 대 B2, B4 대 B1의 지도 입력 비교. 보조 과제의 보편적 우위 보장 없음.'],
        ['확률 0.5의 임의 경고 적용','별도 보정 비행에서 온도와 두 경고 기준 결정','보정의 목적: 확률 해석과 경고 운용점 설정. AP 순위 개선과 구분.'],
        ['잔여 수색의 왕복·대기 반복','선두 목표 예약, 우선권 보존, 반복 시 안전 양보·임시 경유','전체 수색 완료 경로 71/90 → 90/90. 통신 위험 기반 재배치와 별도.']])+ 
        '<p>이번 비교의 범위: 동일 Transformer 계열 안의 입력·학습 방식 개선. GRU·TCN보다 Transformer가 항상 우수하다는 결론 미포함.</p>','reason')
    rows=[]
    for name,v in comp['results'].items():
        if not name.endswith('_s11'):continue
        b=v['balanced']['all'];variant=name.rsplit('_s',1)[0];rows.append([LABELS.get(variant,variant),num(b['average_precision']),num(v['balanced']['macro_region_ap']),pct(b['precision']),pct(b['recall']),pct(b['false_positive_rate']),f'{v["parameters"]:,}'])
    for key,label in [('current_sinr_baseline','현재 SINR 단독 · 비학습'),('geometry_extrapolation_baseline','지도 등속 외삽 · 비학습')]:
        if comp.get(key):
            b=comp[key]['balanced']['all'];rows.append([label,num(b['average_precision']),num(comp[key]['balanced']['macro_region_ap']),pct(b['precision']),pct(b['recall']),pct(b['false_positive_rate']),'—'])
    replicas=[]
    for seed in (11,22,33):
        v=comp['results'][selected.rsplit('_s',1)[0]+f'_s{seed}'];b=v['balanced']['all'];c=v['conservative']['all'] if v['conservative'] else None
        replicas.append([str(seed),num(b['average_precision']),num(v['balanced']['macro_region_ap']),pct(c['precision']) if c else '—',pct(c['recall']) if c else '—'])
    body+=block('3. 모델 비교와 선택',table(['변형','AP','지역 평균 AP','정밀도','재현율','FPR','파라미터'],rows)+
        f'<p>선정 기준: 검증 선택 비행의 pooled AP 최대. 대표 초기화 시드 11. 선정 모델의 추가 초기화 시드 22·33. 앙상블 없음. 최대 4,000회 업데이트, 배치 256, 학습률 0.0003, 체크포인트 선택 주기 250회.</p>'+
        '<p>비교한 학습 손실: 단일 정답의 BCE(5초), 또는 BCE(5초) + 0.3 × BCE(10초) + 0.1 × Huber(미래 최소 SINR). 계수 0.3·0.1은 프로젝트의 사전 설정값. 논문의 보편적 권장값과 구분.</p>'+
        '<h3>선정 모델의 초기화 반복 · 동일 평가 표본</h3>'+table(['시드','AP','지역 평균 AP','보수 정밀도','보수 재현율'],replicas)+image_tag(figs[2]))
    body+=block('4. 지도 사용 방법과 데이터 분할',table(['항목','설정'],[
        ['지역/기록','45개 지역 × 2방향 × 3간섭 조건 = 270개 전체 비행'],
        ['시뮬레이션 시간',f'{sum(r["metadata"]["duration_s"] for r in manifest["records"])/3600:.2f}시간'],
        ['학습/검증/평가','30개 / 7개 / 8개 지역. 인접·중복 건물 지역의 같은 분할 유지.'],
        ['검증 선택/보정','각 검증 지역의 6개 비행 중 3개/3개 분리. 같은 비행의 시간 창 중복 사용 없음.'],
        ['학습 표본',f'{run["training_samples"]:,}개. 지역별 최대 8,192개. 정답을 보지 않는 균등 추출.'],
        ['예측 입력/정답','과거 20초 → 향후 5초 안의 1초 이상 연속 양방향 링크 불가용'],
        ['수집/결정 주기','무선 계산 0.5초, 이동 계획 3초, 예측 결정 1초'],
        ['정답용 가용 링크','데이터 + 역방향 ACK의 홉 성공 확률 0.9 이상을 양방향 모두 충족'],
        ['추가 지도 입력','비콘 상태의 등속 외삽. 현재/3초/5초 손실·LoS와 외삽 비행선의 장애물 교차'],
        ['고정 배치','총 8대·수색 3대·중계 5대. 명목 수색 V 대형 40m. 배치 최적화 재실험 미실시.']])+
        '<p>여기서 단절: 위 가용 링크 기준의 미충족. 실제 패킷 전달률 0 또는 군집 전체 네트워크의 분리와 구분.</p>'+
        '<p>모델 입력의 금지 항목: 실제 미래 궤적, 미래 채널, 간섭원 위치·활성 시각, 정답용 현재 링크 가용성. 평가 집단의 정의에 사용한 정답용 현재 가용성과 실행 입력의 구분.</p>','data')
    op=[]
    for key,label in [('balanced','일반 경고'),('conservative','보수 경고')]:
        v=runtime[key+'_windows'];e=runtime[key+'_events'];a=runtime['alarm_onsets'][key]
        if v:op.append([label,pct(v['precision']),pct(v['recall']),pct(v['false_positive_rate']),pct(e['all_event_recall_at_least_3s']),pct(a['onset_precision']),num(a['false_or_late_onsets_per_drone_hour'],2)])
    body+=block('5. 실제 실행 조건의 위험 점수와 알림',table(['점수 기준','경고 창 정밀도','재현율','FPR','모든 단절 중 3초 이상 경고','첫 알림 정밀도','드론·시간당 오경보/늦은 첫 알림'],op)+
        '<p>운용 후보: 20초 기록, 과거 비콘 수신 3회 이상, 양방향 측정 나이 2초 이하, 측정상 현재 통신 여유 양수. 정답용 현재 가용성 미사용. 위 표와 1절의 오프라인 가용 링크 평가 집단의 차이.</p>'+ 
        '<p>첫 알림 정밀도: 연속 경고의 시작 시각을 기준으로 향후 5초 구간에 새 단절의 1초 이상 지속 기준까지 포함된 비율. 이미 진행 중인 단절의 늦은 알림에 대한 정답 인정 없음. 창 단위 정밀도와 구분.</p>'+ 
        '<p>측정 여유: 양방향 관측 SINR 중 작은 값 − 약 9.21dB. 현재 BPSK·패킷 길이의 대칭 홉 성공 확률 0.9에 대응하는 모델 계산값. 실제 장비의 보편적 임계치와 구분.</p>'+ 
        table(['운용 설정','값·판단 기준'],[
            ['일반 문턱',num(cal['threshold_balanced'],5)+' · 검증 보정 표본의 FPR 5% 이하'],
            ['보수 문턱',num(cal['threshold_conservative'],5)+' · 검증 보정 표본의 FPR 2% 이하 및 정밀도 70% 이상'],
            ['목표 기준의 출처','프로젝트 운용 목표. 문헌의 성공 판정 표준값 아님. 평가 자료에서의 달성 보장 없음.'],
            ['확률 보정 온도',num(cal['temperature'])],
            ['보정 전/후 Brier · 보정 표본',num(cal['brier_before'],5)+' / '+num(cal['brier_after'],5)],
            ['관측 부족·노후','unknown 상태. 낮은 위험 0으로 대체 없음.'],
            ['현재 품질 저하','현재 저하 상태. 사전 경고와 별도 상태값.'],
            ['점수 기준의 알림 중복','연속 경고의 첫 알림 표시. 임의 쿨다운·경고 지연 추가 없음.']])+ 
        '<h3>기본 알림의 보완 · 높은 진입 문턱과 낮은 해제 문턱</h3>'+ 
        '<p>기존 점수 기준의 첫 알림 과다를 확인한 뒤 검증 보정 비행만으로 알림 정책 설계. 목표: 첫 알림 정밀도 80% 이상, 드론·시간당 오경보 5회 이하. 목표 안에서 3초 이상 선행 첫 알림 비율 최대화. 문헌의 보편적 기준과 구분.</p>'+ 
        table(['설정','선정값'],[
            ['알림 진입/해제',num(alarm['selected']['enter_threshold'],5)+' / '+num(alarm['selected']['exit_threshold'],5)],
            ['상태 규칙','진입 문턱 이상에서 경고 시작, 해제 문턱 미만에서 종료. 관측 부족·노후·현재 저하 시 초기화.'],
            ['신경망 점수','원래 연속 위험 점수 유지. 알림 상태 처리에만 히스테리시스 적용.'],
            ['추가 경고 대기','0초. 높은 문턱에 따른 늦은 탐지와 별도.']])+ 
        '<h3>정책 고정 뒤의 새 난수 8개 비행</h3>'+ 
        table(['항목','이전 보수 점수 기준','기본 고신뢰 알림'],[
            ['첫 알림 정밀도',pct(confirmation['previous_point_warning']['first_alert_precision']),pct(confirmation['new_alarm_policy']['first_alert_precision'])],
            ['드론·시간당 오경보/늦은 첫 알림',num(confirmation['previous_point_warning']['false_alerts_per_drone_hour'],2),num(confirmation['new_alarm_policy']['false_alerts_per_drone_hour'],2)],
            ['3초 이상 선행 첫 알림으로 포착한 새 단절',pct(confirmation['previous_point_warning']['first_alert_event_recall_3s'],2),pct(confirmation['new_alarm_policy']['first_alert_event_recall_3s'],2)],
            ['경고 유지 중 3초 이상 앞서 포착한 새 단절',pct(confirmation['previous_point_warning']['event_recall_3s'],2),pct(confirmation['new_alarm_policy']['event_recall_3s'],2)],
            ['경고 창 재현율',pct(confirmation['previous_point_warning']['warning_window_recall']),pct(confirmation['new_alarm_policy']['warning_window_recall'])]])+ 
        f'<div class="callout"><b>고신뢰 알림의 역할: 오경보가 적은 임박 위험 통보.</b><p>새 난수 확인의 3초 이상 선행 첫 알림 포착률 {pct(confirmation["new_alarm_policy"]["first_alert_event_recall_3s"],2)}. 충분한 사전 재배치 시간 확보와 미충족. 재배치 학습에 제공할 연속 위험 점수와 고신뢰 알림의 용도 구분.</p></div>'+ 
        '<p>추가 8개 비행은 정책을 고정한 뒤 평가. 기존 48개 비행의 정책 재적용 결과는 탐색적 재검증. 두 결과의 구분과 처음 보는 지역에 대한 일반화 확인 필요.</p>','operation')
    body+=block('6. 시뮬레이터와 통신 비용',table(['계층','현재 구현'],[
        ['공간','400m × 400m. 건물 폴리곤·높이. 고도 50m. 50m 이상 건물 + 수평 5m 안전 영역의 비행 제한.'],
        ['채널','2.4GHz·송신 10dBm·대역폭 1MHz. ITU 자유공간 손실 + 지붕 프로파일의 Bullington 회절 근사.'],
        ['시간 변화','표준편차 1.5dB·상관시간 4초의 섀도잉. 외부 간섭원 1개·높이 2m·출력 −3~3dBm.'],
        ['간섭 조건','간섭 없음 / 임무 내내 / 임무의 일부 구간. 위치·구간의 난수 시드 고정. 예측 입력에서 제외.'],
        ['성공 확률','비부호화 coherent BPSK와 독립 비트 오류 가정. 실제 무선 칩 실측 곡선과 구분.'],
        ['MAC/비콘','이상적 TDMA 5ms. 드론별 매초 128B 브로드캐스트, ACK 없음. 8슬롯/초 예약.'],
        ['비콘 측정','성공 수신 시 SINR·수신 전력 측정 및 상대 상태 갱신. 측정 오차 표준편차 0.5dB 가정.'],
        ['라우팅','전역 링크 정보를 사용하는 기대 전송 횟수 비용 경로. 실제 분산 라우팅 스택 구현 및 전역 정보 교환 비용 미포함.'],
        ['3GPP/ns-3','TR 36.777 계수 및 ns-3 엔진 미사용. 기존 물리 채널 원코드의 해시 일치 확인.']])+ 
        '<p>계산 흐름: 거리·건물 → 자유공간 손실 + 회절 손실 → 수신 전력 → SINR → 비트 오류율 → 데이터·ACK 성공 확률. SINR: 수신 신호 전력 / (간섭 전력 + 잡음 전력), 선형 전력 기준.</p>'+ 
        f'<p>비콘 전용 슬롯 비율: {pct(cost["reserved_slot_fraction"])}. 1Mbps의 비콘 프레임 직렬화 시간 비율: {pct(cost["beacon_rf_airtime_fraction"],3)}. 예약 슬롯과 실제 프레임 길이의 구분.</p>'+ 
        f'<p>고정 18개 비행의 슬롯 예약 유무 비교: 평균 PDR 차이 {100*cost["average_pdr_difference"]:+.3f}%p, 평균 전달 지연 차이 {1000*cost["mean_delay_difference_s"]:+.2f}ms. 같은 저장 채널과 초기 난수 시드. 송신 시점 변경에 따른 패킷별 난수 순서 차이 반영.</p>'+ 
        f'<p>수색 경로 확인: {motion["complete_count"]}/90개 전체 완료. 최소 연속 분리 거리 {motion["minimum_separation_m"]:.2f}m. 수색용 반복 탈출 중 명목 대형 변형 허용. 통신 예측을 이용한 재배치와 별도.</p>','sim')
    body+=block('7. 바로 확인하는 사용 방법',table(['목적','파일·실행'],[
        ['설치 없는 결과 확인','현재 HTML과 통신위험_전체임무_재생.html 열기'],
        ['Windows 환경','00_환경설정.bat. Python 3.12, 프로젝트 가상환경 1개, CPU PyTorch. 운영체제 변경·가상 머신 불필요.'],
        ['실제 예측기 재생','01_예측기_재생.bat. 포함된 3개 전체 비행에서 observe 인터페이스 실행.'],
        ['환경 확인','python usable_forecast.py doctor'],
        ['전체 데이터 재생성','python usable_forecast.py collect --workers 2'],
        ['학습 자료 준비/학습','python usable_forecast.py prepare → python usable_forecast.py train --workers 2'],
        ['전체 평가','python usable_forecast.py evaluate'],
        ['모듈 연결','usable.predictor.LinkRiskPredictor.observe / observe_local'],
        ['기본 입력','0.5초 간격 N×N×48 값과 관측 마스크. 정적 지도 경로 제공 시 12개 지도 입력의 내부 생성.'],
        ['출력','링크별 5초 위험 점수, 정상·주의·경고·관측 부족·관측 노후·현재 저하, 첫 경고 여부.'],
        ['드론별 실행','observe_local의 자기 관측 행만 사용. 다른 드론의 비공개 관측 행 미요구.'],
        ['추가 자료','results/usable의 JSON·CSV, 학습 이력, 전체 기록 식별자와 코드 해시. 원시 기록의 별도 분할 압축 파일.']]),'use')
    body+=block('8. 기타 설명',bullets([
        'AP: 위험 순위의 정밀도–재현율 성능 요약. 값 0~1. 정확도 또는 연결 유지율과 다른 지표.',
        'pooled AP: 모든 평가 창의 통합 계산. 지역 평균 AP: 지역별 AP를 같은 비중으로 평균. 긴 임무의 영향 차이.',
        'FPR: 실제 정상인 창 중 경고한 비율. 경고 중 오경보 비율과 다른 분모.',
        '재현율: 실제 미래 위험 창 중 경고한 비율. 경고 정밀도: 경고한 창 중 실제 미래 위험 비율.',
        'Brier 점수: 예측 확률과 정답 사이의 평균 제곱오차. 작은 값이 유리.',
        '사전 경고 3초: 새 단절 시작보다 최소 3초 앞선 유효 경고. 관측 가능한 사건과 모든 새 사건의 분모 구분.',
        '양방향 링크의 두 관측 방향은 상관된 표본. 시간 창과 방향 링크를 독립 반복 실험처럼 취급한 신뢰구간 미사용.',
        '지도와 전파 함수의 학습·평가 가정 공유. 건물 높이 추정 오차, 실제 무선 장비, 비동기 채널 접근에 대한 성능 검증 미실시.',
        '모델 학습용 PC와 비행 중 추론의 분리. 비행 중 재학습 미요구. 중앙 관제소 없는 로컬 링크 추론 인터페이스.',
        '실제 이동을 바꾸는 재배치 정책 연결 후의 궤적 분포 변화와 통신 유지 효과에 대한 별도 검증 필요.',
        '3초 이동 주기와 요구 회피 이동 시간을 반영한 선행시간 검토 필요. 10초 보조 출력의 정식 평가 미실시.',
        '다음 개선의 원인 후보: 3초 이동 명령 이후의 조향 의도 미입력, 임박 장애와 3초 이후 장애를 합친 현재 정답. 원인 확정 실험 미실시.',
        '다음 개발 제안: 3~5초 뒤 새 단절의 별도 정답, 예정 경유점·계획 속도의 비콘 공유, 안전한 행동 다양화 데이터, 제어 비용 기반 E2E 비교. 개선 효과의 별도 검증 필요.',
        '신규 지역의 완전한 맹검 일반화 확인 및 다른 드론 대수·대형·고도의 평가 필요.',
        '관측 전까지 징후가 없는 임의 간섭 시작의 완전한 예측 보장 없음.',
        '체크포인트 선택과 경고 문턱 설정은 학습·검증 영역에서 완료. 평가 결과에 맞춘 문턱 재설정 없음.'
    ]),'limits')
    refs=[
        ('RFC 6130 · NHDP','https://www.rfc-editor.org/rfc/rfc6130.html','§4.3.1, §5.3.2, §5.3.3. 주기적 HELLO, 정보 유효 시간, 서로 다른 진입·해제 문턱의 원리 참고. 1Hz·128B·4%와 알림 문턱은 프로젝트 설정. NHDP 전체 구현 아님.'),
        ('Guo et al. · On Calibration of Modern Neural Networks','https://proceedings.mlr.press/v70/guo17a.html','ICML 2017, §4. 온도 스케일링의 검증 자료 기반 확률 보정. 본 프로젝트의 AP 향상 근거와 구분.'),
        ('Caruana · Multitask Learning','https://www.cs.cornell.edu/~caruana/mlj97.pdf','Machine Learning 1997, §1.3. 관련 과제의 공유 표현과 복수 출력 원리. UAV 데이터의 우위 증명 자료 아님.'),
        ('Okumura et al. · Priority Inheritance with Backtracking','https://arxiv.org/html/1901.11282v5','§4. 우선순위 상속·되돌림 원리. 본 구현의 연속 분리거리·목표 예약·반복 탈출은 프로젝트 변형. 원 논문의 도달 보장 조건과 구분.'),
        ('ITU-R P.525 / P.526','https://www.itu.int/rec/R-REC-P.525/en','자유공간 손실 및 회절 원리. P.526-16 §4.5.1의 단거리 평지 지붕 근사 활용. 표준 전체 방법의 구현과 구분.')]
    body+=block('9. 구현 근거와 원문 확인 범위',table(['자료','이번 구현과의 연결'],[[f'<a href="{url}" target="_blank" rel="noopener">{label}</a>',desc] for label,url,desc in refs]))
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f7;color:#19293e;font:15px/1.75 Arial,"Noto Sans KR","Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:28px 24px 60px}.hero{background:#142b46;color:white;padding:42px;border-radius:18px}.eyebrow{font-size:11px;color:#9bc7f5;letter-spacing:.18em}h1{font-size:36px;line-height:1.35;margin:15px 0}.hero p{color:#b9cbe0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:25px}.metric{border:1px solid #3a536e;border-radius:10px;padding:19px 15px}.metric b{display:block;font-size:30px;color:#fff}.metric span{font-size:11px;color:#b8cbe0}nav{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0}nav a{background:white;padding:8px 15px;border-radius:8px;border:1px solid #dde5ee;font-size:12px;text-decoration:none;color:#345b83}section{background:white;border:1px solid #dce4ef;border-radius:14px;padding:28px;margin-top:20px;scroll-margin-top:20px}h2{font-size:22px;margin:0 0 20px}p,li{font-size:13px}p{margin:13px 0}ul{padding-left:20px}li{margin-bottom:8px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.7}th{text-align:left;background:#edf3fa;color:#335372;padding:12px 13px;border-bottom:2px solid #d4e0ed;white-space:nowrap}td{padding:12px 13px;border-bottom:1px solid #e7edf5;vertical-align:top}td:first-child{font-weight:700;color:#2a4564}tr:nth-child(even){background:#fbfcfe}.callout{background:#edf5ff;border-left:4px solid #2765b1;border-radius:6px;padding:15px 18px;margin-bottom:20px}.callout p{margin:5px 0 0}.caption{color:#657b94;font-size:11px}.figure{width:100%;height:auto;margin:18px 0 6px;border:1px solid #e4eaf2;border-radius:10px}a{color:#1a5ca7}code{font-size:12px;background:#edf2f8;padding:3px 6px;border-radius:4px}@media(max-width:800px){main{padding:12px}.hero{padding:25px}h1{font-size:27px}.metrics{grid-template-columns:1fr 1fr}.metric b{font-size:25px}section{padding:20px}h2{font-size:19px}}@media print{body{background:white}main{max-width:none;padding:0}nav{display:none}.hero{border-radius:0}section{break-inside:avoid;border-radius:0}.figure{max-height:400px;object-fit:contain}}'''
    result='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transformer 예측 개선 결과</title><style>'+css+'</style><main>'+body+'</main></html>'
    (out/'Transformer_운용개선_결과.html').write_text(result,encoding='utf-8')
    # 숫자의 단일 출처: 보고서 핵심 수치를 별도 JSON으로 보존합니다.
    summary={'selected_run':selected,'dense_selected_ap':m['average_precision'],'dense_baseline_ap':old['average_precision'],
             'runtime_conservative_window_precision':tight['precision'] if tight else None,
             'runtime_conservative_all_event_recall_3s':rte['all_event_recall_at_least_3s'] if rte else None,
             'motion_completed':motion['complete_count'],'records':len(manifest['records']),
             'confirmation_ap':confirmation['classification']['all']['average_precision'],
             'confirmation_first_alert_precision':confirmation['new_alarm_policy']['first_alert_precision'],
             'confirmation_false_alerts_per_drone_hour':confirmation['new_alarm_policy']['false_alerts_per_drone_hour'],
             'confirmation_first_alert_3s_recall':confirmation['new_alarm_policy']['first_alert_event_recall_3s'],
             'test_geography_status':comp['test_geography_status']}
    write_json(RESULT/'report_summary.json',summary)
    return summary


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
