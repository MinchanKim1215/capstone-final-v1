"""Korean, evidence-bound review PDF. Layout overflow is an error."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import math
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Frame, Flowable

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
D=json.loads((OUT/'interim_evidence/snapshot.json').read_text(encoding='utf-8'))
PDF=OUT/'캡스톤 디자인 II 중간보고서_20261029.pdf'
pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KR-Bold','C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('KR',normal='KR',bold='KR-Bold',italic='KR',boldItalic='KR-Bold')
NAVY=colors.HexColor('#12324A');TEAL=colors.HexColor('#087D86')
GRAY=colors.HexColor('#5D6D7C');LIGHT=colors.HexColor('#EFF5F7')
AMBER=colors.HexColor('#95520F');RED=colors.HexColor('#A33D40')
PAGE_W,PAGE_H=A4
LEFT=49;WIDTH=PAGE_W-98
CUTOFF=D['as_of'][:16].replace('T',' ')+' KST'
MAIN_END=D['main_status'].get('ended_at') or D['as_of']
MAIN_SECONDS=(datetime.fromisoformat(MAIN_END)-datetime.fromisoformat(D['main_status']['started_at'])).total_seconds()
BODY=ParagraphStyle('body',fontName='KR',fontSize=9.4,leading=14.4,wordWrap='CJK',spaceAfter=7,textColor=NAVY)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=8.0,leading=11.8,textColor=GRAY,spaceAfter=5)
H2=ParagraphStyle('h2',parent=BODY,fontName='KR-Bold',fontSize=12.2,leading=18,spaceBefore=11,spaceAfter=8)
CELL=ParagraphStyle('cell',parent=BODY,fontSize=8.2,leading=12,spaceAfter=0)
HEAD=ParagraphStyle('head',parent=CELL,fontName='KR-Bold',textColor=colors.white)
BOX=ParagraphStyle('box',parent=BODY,fontSize=9.4,leading=14.6,spaceAfter=0)
SOURCE=ParagraphStyle('source',parent=SMALL,fontSize=7.5,leading=10.6)


def clean(text):
    for character in ['\u2010','\u2011','\u2012','\u2013','\u2014','\u2212']:
        text=text.replace(character,'-')
    return text


def p(text,style=BODY):
    return Paragraph(clean(text),style)


def text(value):
    return escape(str(value))


def h(text):
    return p(text,H2)


def note(label,body,color=TEAL):
    result=Table([[p('<b>'+text(label)+'</b><br/>'+body,BOX)]],colWidths=[WIDTH])
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('LINEBEFORE',(0,0),(0,-1),3,color),
                               ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
                               ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
    return result


def table(headers,rows,widths=None):
    if widths is None:
        widths=[WIDTH/len(headers)]*len(headers)
    else:
        total=sum(widths);widths=[WIDTH*w/total for w in widths]
    data=[[p(text(v),HEAD) for v in headers]]+[[p(text(v),CELL) for v in row] for row in rows]
    result=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
                               ('LINEBELOW',(0,0),(-1,0),.6,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DDE5E9')),
                               ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                               ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return result


def refs(ids,extra=''):
    return p('근거 '+', '.join('['+q+']' for q in ids)+('. '+extra if extra else ''),SOURCE)


def pct(value,digits=3):
    return f'{value*100:.{digits}f}%'


def signed(value,digits=3):
    return f'{value:+.{digits}f}'


class LineChart(Flowable):
    def __init__(self,points,baseline):
        super().__init__();self.width=WIDTH;self.height=164;self.points=points;self.baseline=baseline
    def draw(self):
        c=self.canv;x0=42;y0=28;w=WIDTH-58;hh=112;low=96.;high=99.
        c.setFont('KR',7.2)
        for tick in [96,97,98,99]:
            y=y0+(tick-low)/(high-low)*hh
            c.setStrokeColor(colors.HexColor('#DFE7EB'));c.line(x0,y,x0+w,y)
            c.setFillColor(GRAY);c.drawRightString(x0-8,y-3,f'{tick:.0f}%')
        for tick in [0,64,128,192,256]:
            x=x0+tick/256*w;c.setFillColor(GRAY);c.drawCentredString(x,12,str(tick))
        c.setStrokeColor(AMBER);c.setDash(4,3)
        y=y0+(self.baseline*100-low)/(high-low)*hh;c.line(x0,y,x0+w,y);c.setDash()
        c.setStrokeColor(TEAL);c.setLineWidth(1.6)
        previous=None
        for update,value in self.points:
            x=x0+update/256*w;y=y0+(value*100-low)/(high-low)*hh
            if previous:c.line(*previous,x,y)
            c.setFillColor(TEAL);c.circle(x,y,2.4,fill=1,stroke=0)
            if update==176:
                c.setFillColor(NAVY);c.drawString(x-22,y+10,'best 176')
            previous=(x,y)
        c.setFillColor(GRAY);c.drawString(x0,151,'검증 CTR - 세로축 96~99%, 가로축 PPO update')


class APBars(Flowable):
    def __init__(self,rows):
        super().__init__();self.width=WIDTH;self.height=142;self.rows=rows
    def draw(self):
        c=self.canv;x0=128;w=WIDTH-180;top=115
        for index,(label,value) in enumerate(self.rows):
            y=top-index*28
            c.setFillColor(NAVY);c.setFont('KR',8.1);c.drawRightString(x0-10,y+2,label)
            c.setFillColor(LIGHT);c.rect(x0,y,w,13,fill=1,stroke=0)
            c.setFillColor(TEAL if index<2 else NAVY);c.rect(x0,y,w*value/.55,13,fill=1,stroke=0)
            c.drawString(x0+w+9,y+2,f'{value:.4f}')
        c.setFillColor(GRAY);c.setFont('KR',7.2);c.drawString(x0,6,'공통 관측 5초 위험 AP / 가로축 0~0.55')


pages=[]


def add(title,subtitle,story):
    pages.append((title,subtitle,story))


comm=D['communication'];means={name:value['means'] for name,value in comm.items()}
base=means['physics'];old=means['original_joint'];fixed=means['original_fixed'];new=means['best176']
common=D['common']['common'];pred=D['latest']['joint']['summary']['prediction'];alarms=D['latest']['joint']['summary']['alarms']
updates=D['progress']['committed_updates'];pilot=D['pilot_status'];cases=8
full=D['latest']['joint'];full_base=D['latest']['physics'];legacy=D['legacy_html']
if pilot.get('state')=='completed':
    pilot_text=f'예측 학습 6,000단계와 8관측 재생 완료. best {D.get("pilot_results",{}).get("best_step",5500):,}단계'
elif pilot.get('stage')=='full_prediction_replay':
    pilot_text=f'예측 학습 6,000단계 완료. best {pilot.get("best_step",5500):,}단계의 전체 재생 중'
elif 'step' in pilot:
    pilot_text=f'예측 전용 학습 {pilot["step"]:,}/{pilot.get("target",6000):,}단계'
else:
    pilot_text=f'자료 {pilot.get("complete",0)}/200회 준비'

add('캡스톤 디자인 II 중간보고서','이전 HTML·배포 v5·현재 본 학습의 통합 비교 | '+CUTOFF,[
    note('핵심 판정','본 학습 256회와 정책별 전체 96조건 검증 완료. 같은 96조건에서 통신 유지 성능 개선 확인.<br/>이전 예측 전용 HTML보다 예측 성능이 향상됐다는 동일 조건 증거는 없음. 조기 사건 탐지는 여전히 미흡.'),
    Spacer(1,14),
    table(['항목','기준 시점의 확인 결과'],[
        ['코드 기준','통합 ZIP과 핵심 파일 966개 일치. Python 소스 161개 일치. 같은 v5.'],
        ['본 학습',f'공동 PPO {updates}/256 update 완료. 훈련 비행 {D["progress"]["training_episodes"]:,}회. 전체 검증에서 176 선택.'],
        ['전체 96조건 통신',f'physics 대비 CTR {pct(full_base["means"]["search_connected_mean"])} → {pct(full["means"]["search_connected_mean"])}. 평균 단절 {full_base["means"]["outage_total_s"]:.3f} → {full["means"]["outage_total_s"]:.3f}초.'],
        ['동일 관측 예측',f'5초 위험 AP {common["pretrain1200"]["risk_5s"]["average_precision"]:.4f} → {common["joint176"]["risk_5s"]["average_precision"]:.4f}. 정확도 백분율이 아닌 점수.'],
        ['이전 HTML과 예측',f'과거 48비행 AP {legacy["ap"]:.3f}, 현재 96비행 AP {pred["risk_5s"]["average_precision"]:.4f}. 평가 집단이 달라 단순 차이를 개선량으로 해석하지 않음.'],
        ['조기 경보 한계',f'전체 96조건의 8초 경보: 정밀도 {pct(alarms["8s"]["first_alert_precision"],2)}, 3초 선행 사건 재현율 {pct(alarms["8s"]["first_alert_event_recall_3s"],3)}.'],
        ['현재 개선 실험',pilot_text+'. 새 정답의 별도 실험이며 최종 개선 효과 판정 전.'],
    ],[1.1,3.8]),
    h('읽기 기준'),
    p('① 이전 usable HTML의 48비행. ② 통합본 v5와 현재 모델의 고정 8관측. ③ 초기 8조건 정책 비행. ④ 전체 96조건 정책 비행. 네 집합을 분리함. 같은 조건의 변화와 서로 다른 기록의 단순 차이를 구분함.'),
    p('수치는 위 기준 시점에 고정함. 이후 진행되는 학습 상태는 본 PDF에 자동 반영되지 않음. 모든 결과는 시뮬레이션 내부 검증임.',SMALL),
    refs(['D01','D05','D14','D19','L01','L02','L03','U01'])
])

add('01. 코드 기준과 변경 범위','용어 | 통합본: 파일 묶음 / checkpoint: 특정 시점의 저장 가중치',[
    table(['구분','중간보고 통합본','현재 컴퓨터'],[
        ['알고리즘 코드','단독 Transformer + 공동 PPO v5','핵심 불변 파일 966개와 동일'],
        ['Python 소스','161개 대조','불일치 0개, 누락 0개'],
        ['설정·고정 모델','configs 23개, 고정 .pt 108개 포함','대조 항목의 바이트 지문 일치'],
        ['자료 범위','기존 본 ZIP + 추가 원시 기록 44개 + 인계 문서','이후 지도학습·PPO 비행·평가 기록 추가'],
        ['가중치 시점','과거 배포 시점 상태','이후 본 학습 checkpoint 추가'],
        ['예측 개선 코드','새 구간 예측 기능 미포함','prediction_improvement/에 별도 구현'],
    ],[1.05,1.9,2.25]),
    h('무엇이 더 발전했는가'),
    p('통합본은 실행 증거와 인계 자료가 더 완전한 패키지임. 새 알고리즘 버전은 아님. 현재 PC는 동일 알고리즘으로 더 학습한 상태임. 가중치가 최신이라는 사실만으로 모든 예측 지표가 더 좋다고 판단할 수 없음.'),
    h('원본 보존'),
    p('통합 ZIP을 진행 중인 학습 폴더에 덮어쓰지 않음. 보상·지도 분할·안전 이동기·기존 PPO 코드를 변경하지 않음. 예측 개선은 별도 폴더와 실험 계약으로 관리함.'),
    p('원본 코드·configs 파일의 일치와 실행 시 학습 설정의 일치는 구분함. 새 본 학습은 배치 비행 수·최대 epoch·검증 간격이 과거 소규모 run과 다름. 다음 표에 별도 기록함.',SMALL),
    note('판단 근거','문서 설명뿐 아니라 ZIP 내부 파일과 현재 폴더의 SHA256을 비교함. 파일 크기나 ZIP 안의 물리적 파일 수로 버전을 판단하지 않음.'),
    p('SHA256: 파일 내용의 동일성을 점검하는 암호학적 지문. joint: 예측기·정책을 함께 갱신하는 모드. fixed: 예측기를 고정하고 정책을 학습하는 모드.',SMALL),
    refs(['D01','D02','D03','C01','C02'])
])

add('02. 현재 알고리즘의 연결 구조','용어 | 예측: 미래 위험 추정 / 재배치: 중계 드론의 이동 선택',[
    table(['단계','실제 구성','역할'],[
        ['관측','0.5초 RF 40프레임 = 과거 약 20초','링크별 통신·위치·의도 이력'],
        ['입력','값 48 + 마스크 48 + 의도 9 = 105차원/시점','결측 구분, 물리량 스케일링'],
        ['지도·물리','현재 수신 정보로 계산한 46차원 특징','알려진 지도·의도에 따른 차폐·통신 여유'],
        ['Transformer','hidden 32, attention head 4, encoder 2층','시간 이력의 표현과 4개 위험 확률 출력'],
        ['공동 PPO','공유 actor + 학습 전용 중앙 critic','중계 5대의 안전 후보 25개 중 이동 선택'],
        ['안전 이동','총 8대: 수색 3대 + 중계 5대','장애물·분리거리·속도 제한 적용'],
    ],[.8,2.4,2.0]),
    h('운용과 학습의 구분'),
    p('중앙 critic은 훈련용임. 실제 운용에 지휘소/GCS를 새로 가정한 구조가 아님. 기본 예측·정책 입력에 GRU나 LightGBM 점수를 사용하지 않음.'),
    h('예측이 제어에 전달되는 방식'),
    p('PPO는 위험 확률과 시계열 표현을 연속 값으로 사용함. 경보 문턱 0.5를 넘을 때만 이동하는 구조가 아님. 따라서 경보 재현율이 낮아도 제어에 도움이 될 가능성은 있음. 기여의 크기와 인과관계는 고정 예측기·공동학습의 통제 비교가 추가로 필요함.'),
    p('encoder: 이력을 특징 벡터로 바꾸는 부분. head: 목표별 예측 출력을 내는 부분. actor: 행동을 선택하는 정책. critic: 학습 시 가치를 평가하는 망.',SMALL),
    note('예측 정답의 의미','기록된 정책을 실행했을 때 실제로 발생한 단절임. 재배치하지 않았다면 발생했을 위험이라는 반사실 정답은 아님.'),
    refs(['C01','D20','R03'])
])

add('03. 실제 학습 진행 이력','용어 | step: 최적화 단계 / episode: 비행 1회 / update: PPO 갱신 묶음',[
    table(['항목','통합본 상태','현재 본 학습 상태'],[
        ['지도학습용 훈련 비행','24회','192회. 기존 24회에서 168회 추가'],
        ['훈련 지역','166개 중 일부','고정 corpus가 166개 훈련 지역 포함'],
        ['고정 검증 관측','8회, 서로 다른 8개 지역','동일 8회 유지'],
        ['사전학습','누적 1,200 step, best 1,200','누적 3,000 step 완료, best 2,800'],
        ['공동 PPO','verified_joint: 2 update, 8회 훈련','새 transformer_main: '+str(updates)+' update, '+f'{updates*8:,}회 훈련'],
        ['update당 학습 설정','비행 4회, 최대 2 epoch','비행 8회, 최대 4 epoch. minibatch 16 동일'],
        ['검증 간격 / 조기 중단','매 update / patience 0','16 update / 검증 8회 정체 시 중단'],
        ['본 학습 정기 검증','소규모 검증 기록',f'16~{D["validation_history"][-1]["update"]} update의 {len(D["validation_history"])}개 평가 시점'],
        ['제어 best','verified_joint 1 / verified_fixed 1','transformer_main 176'],
    ],[1.2,1.7,2.5]),
    p('epoch는 수집 표본을 학습하는 반복, minibatch는 한 번의 최적화에 쓰는 표본 묶음임. 본 update의 8회 비행과 최대 4 epoch는 서로 다른 단위임.',SMALL),
    Spacer(1,8),
    LineChart([(row['update'],row['score'][0]) for row in D['validation_history']],base['search_connected_mean']),
    p('점선: physics 기준선. 실선: 동일 8조건의 정기 검증 CTR. 최신 update가 항상 best는 아님. best는 안전·완료·비행시간 조건을 통과한 뒤 CTR, PDR, 최장 단절 순서로 선택함.',SMALL),
    p('본 학습은 verified_joint의 update 2에서 단순 재개한 실행이 아님. 사전학습 best와 기존 제어 기반 모델로 새 transformer_main을 시작함. 256 update와 정책당 96조건 검증을 완료함.',SMALL),
    p('본 실행: 2026-10-29 04:35 시작 → '+MAIN_END[:16].replace('T',' ')+' KST 종료. 약 '+str(int(MAIN_SECONDS//3600))+'시간 '+str(int(MAIN_SECONDS%3600//60))+'분. 종료 코드 0. 자료 수집·사전학습·PPO·검증을 포함한 경과 시간이며 GPU만 계산한 시간은 아님.',SMALL),
    refs(['D03','D05','D10','D11','D12','D13','D23','D25'])
])

add('04. 학습 방식과 검증 범위','용어 | 지도학습: 정답으로 학습 / 강화학습: 행동 결과의 보상으로 학습',[
    h('두 종류의 학습'),
    table(['구분','학습 대상','목적과 해석'],[
        ['사전 지도학습','Transformer encoder와 위험 출력층','미래 링크 단절 정답에 BCE 손실 적용'],
        ['공동 PPO','재배치 정책 + encoder + 위험 출력층','통신·임무 보상과 예측 보조 손실을 함께 사용'],
        ['예측 보조계수 0.1','현재 공동 PPO 총손실의 한 계수','예측을 전체의 10%만 학습한다는 뜻이 아님'],
        ['best 선택','제어 best와 예측 best의 기준이 다름','PPO best 176을 예측 최적 모델로 단정하지 않음'],
    ],[1.1,1.8,2.4]),
    p('PPO: 정책의 급격한 변화를 제한하며 보상으로 갱신하는 강화학습 방법. BCE: 이진 정답과 예측 확률의 차이를 벌점으로 계산하는 손실.',SMALL),
    h('보상과 교수 피드백의 해석'),
    p('기존 통신 우선 보상 구조 유지. 단절 시간, 연속 단절, 패킷 품질, 추가 이동, 수색 지연, 보호·시간·실패 비용을 반영함. 누적 보상 크기를 성능의 직접 근거로 삼지 않고 CTR·PDR·단절·비행시간·안전을 분리 평가함. 현재 학습에서 교수 피드백을 이유로 새 수치 조정을 추가한 사실은 없음.'),
    h('현재까지 확인한 실행 연결'),
    p('Windows NVIDIA CUDA 순전파·역전파·저장·재개 경로 확인. CPU 수집과 GPU 재평가의 행동 확률 일치 확인. 공동 PPO의 정책 gradient가 encoder와 예측 출력층에 전달됨을 기록으로 확인함. 원래 150개 핵심 소스 지문과 수집 자료의 계약을 유지함.'),
    p('실제 장치 검사: RTX 4070 SUPER, PyTorch 2.9.1+cu128. CPU/GPU 목표 log-probability 최대 오차 2.3842e-7. 모델·optimizer 재개 결과의 비트 단위 일치 확인.',SMALL),
    note('평가 범위','초기 비교는 8개 지역의 축소 검증임. 이후 24지역 × 4조건 = 정책당 96조건의 전체 검증을 완료함. 최종 미사용 테스트는 32지역 × 4조건 = 정책당 128조건이며 사용하지 않음.'),
    refs(['C01','D04','D05','D06','D22','D23','D24','R03'])
])

communication_rows=[]
for label,field,mult,unit in [('CTR','search_connected_mean',100,'%'),('PDR','pdr',100,'%'),('지연 반영 PDR','timely_pdr',100,'%'),
                             ('평균 단절 시간','outage_total_s',1,'s'),('비행별 최장 단절의 평균','outage_max_s',1,'s'),
                             ('평균 패킷 지연','delay_mean_s',1000,'ms'),('비행별 지연 p95의 평균','delay_p95_s',1000,'ms'),
                             ('중계 5대 합산 이동','relay_travel_m',.001,'km')]:
    delta=(new[field]-old[field])*mult
    suffix='pp' if unit=='%' else unit
    communication_rows.append([label,f'{base[field]*mult:.3f}',f'{old[field]*mult:.3f}',f'{new[field]*mult:.3f}',signed(delta)+' '+suffix])
add('05. 초기 8조건의 통신·재배치 변화','역사적 축소 검증 | 전체 96조건 결과와 구분함',[
    table(['지표','physics','통합본 joint best','현재 best 176','통합본 joint 대비'],communication_rows,[1.8,1,1.25,1.25,1.5]),
    p('각 값은 8개 비행의 동일 가중 평균임. CTR·PDR 열의 단위는 %. 단절은 초, 지연은 ms, 이동은 km. pp는 퍼센트포인트임. 비행별 p95의 평균은 전체 패킷을 합산한 p95와 다름.',SMALL),
    p('physics는 학습형 단절 예측을 사용하지 않고 물리 후보와 기존 actor로 재배치하는 기준선임. 드론을 움직이지 않는 기준선은 아님.',SMALL),
    table(['별도 항목','physics','통합본 joint','현재 176'],[
        ['8개 비행 전체의 최장 단절',f'{comm["physics"]["longest_outage_s"]:.1f}s',f'{comm["original_joint"]["longest_outage_s"]:.1f}s',f'{comm["best176"]["longest_outage_s"]:.1f}s'],
        ['최대 비행시간 비율',f'{comm["physics"]["max_duration_ratio"]:.4f}',f'{comm["original_joint"]["max_duration_ratio"]:.4f}',f'{comm["best176"]["max_duration_ratio"]:.4f}'],
        ['평균 네트워크 단절 발생 수',f'{base["outage_events"]:.3f}',f'{old["outage_events"]:.3f}',f'{new["outage_events"]:.3f}'],
    ],[2.2,1,1,1]),
    p(f'현재 176: 8/8회 임무 완료·안전 감사 통과. 기록된 안전 위반 {comm["best176"]["safety_violations"]}건. 최소 기체 분리 {comm["best176"]["minimum_separation_m"]:.3f}m. 이 8조건의 시뮬레이션 결과이며 일반적인 안전 보증은 아님.',SMALL),
    h('변화의 크기'),
    p(f'통합본 공동 best 대비 CTR {signed((new["search_connected_mean"]-old["search_connected_mean"])*100)}pp, PDR {signed((new["pdr"]-old["pdr"])*100)}pp. 평균 단절 {old["outage_total_s"]-new["outage_total_s"]:.3f}초 감소({(1-new["outage_total_s"]/old["outage_total_s"])*100:.1f}%).'),
    p(f'physics 대비 CTR {signed((new["search_connected_mean"]-base["search_connected_mean"])*100)}pp. 평균 단절 {(1-new["outage_total_s"]/base["outage_total_s"])*100:.1f}% 감소. 통합본의 선택 정책 fixed best와 비교해도 CTR {signed((new["search_connected_mean"]-fixed["search_connected_mean"])*100)}pp임.'),
    note('이 8조건의 해석',f'이 축소 집합에서는 지연 p95 평균 {(new["delay_p95_s"]/base["delay_p95_s"]-1)*100:.1f}% 증가, 중계 이동 {(new["relay_travel_m"]/base["relay_travel_m"]-1)*100:.1f}% 증가. 전체 96조건에서는 지연 p95 방향이 달라짐. 평가 집합을 구분하여 보고함.',AMBER),
    refs(['D06a','D07a','D08a','D09a'])
])

common_rows=[]
labels=[('pretrain1200','통합본 사전학습 1200'),('pretrain2800','현재 사전학습 best 2800'),('joint176','현재 공동 PPO 176'),('joint190','진단용 공동 PPO 190')]
for name,label in labels:
    metric=common[name]
    common_rows.append([label]+[f'{metric[head]["average_precision"]:.4f}' for head in ['risk_5s','onset_3to4s','onset_3to7s']]+[f'{metric["risk_5s"]["auroc"]:.4f}'])
add('06. 예측 자체의 성능 변화','용어 | AP: 정밀도-재현율의 요약 점수 / AUROC: 위험 점수의 순위 구분 성능',[
    table(['같은 관측에 적용한 모델','5초 위험 AP','3~4초 시작 AP','3~7초 시작 AP','5초 AUROC'],common_rows,[2.3,1,1,1,1]),
    Spacer(1,8),
    APBars([(label,common[name]['risk_5s']['average_precision']) for name,label in labels]),
    h('얼마나 상승했는가'),
    p(f'통합본 사전학습 1200 대비 공동 PPO 176의 5초 위험 AP는 {common["joint176"]["risk_5s"]["average_precision"]-common["pretrain1200"]["risk_5s"]["average_precision"]:+.4f}. 현재 사전학습 2800에서 PPO 176으로의 변화는 {common["joint176"]["risk_5s"]["average_precision"]-common["pretrain2800"]["risk_5s"]["average_precision"]:+.4f}임. AP 증가를 정확도 백분율 증가로 바꾸어 표현하지 않음.'),
    p(f'3~7초 시작 AP는 통합본 {common["pretrain1200"]["onset_3to7s"]["average_precision"]:.4f} → 현재 사전학습 {common["pretrain2800"]["onset_3to7s"]["average_precision"]:.4f} → PPO 176 {common["joint176"]["onset_3to7s"]["average_precision"]:.4f}. 사전학습 이후 공동학습의 추가 개선은 작음.'),
    note('비교 가능 조건','같은 physics 관측 8회, 같은 링크·시점·정답, 같은 공통 지원 마스크를 사용함. 5초 위험 표본 161,341개 중 양성 2,101개(1.302%). 해당 관측은 과거 checkpoint 선택에도 사용되어 독립 최종 검증은 아님.'),
    p('과거 참고값: LightGBM의 같은 관측 5초 AP 0.4736, 교차 적합 단순 혼합 0.4654. 학습 이력은 서로 다름. 현재 비교 작업은 보류하고 예측 기반 개선을 우선함.',SMALL),
    refs(['D14','D15','D23','R01'])
])

add('07. 전체 96조건의 예측·경보 한계','용어 | 표본 재현율과 사건 재현율은 서로 다른 분모를 사용함',[
    table(['best 176의 96조건','5초 위험 출력','3~7초 시작 출력'],[
        ['AP',f'{pred["risk_5s"]["average_precision"]:.4f}',f'{pred["onset_3to7s"]["average_precision"]:.4f}'],
        ['AUROC',f'{pred["risk_5s"]["auroc"]:.4f}',f'{pred["onset_3to7s"]["auroc"]:.4f}'],
        ['문턱 0.5에서 정밀도',pct(pred['risk_5s']['precision'],2),pct(pred['onset_3to7s']['precision'],2)],
        ['문턱 0.5에서 표본 재현율',pct(pred['risk_5s']['recall'],2),pct(pred['onset_3to7s']['recall'],2)],
    ],[2.4,1.2,1.4]),
    p('사건은 5개 중계기의 상대 기체별 유향 링크에서 집계함. 오경보 분모는 중계기들의 총 관찰 시간(드론·시간)임. 네트워크 전체 단절 횟수와 구분함.',SMALL),
    p(f'5초 위험 표본 {pred["risk_5s"]["samples"]:,}개, 양성 {pred["risk_5s"]["positives"]:,}개({pct(pred["risk_5s"]["positive_fraction"],3)}). 고정 8관측 및 초기 자체 8비행의 표본과 다름. 초기 8비행 AP 0.5081과 전체 96비행 AP {pred["risk_5s"]["average_precision"]:.4f}는 같은 가중치에서 평가 집합이 달라진 결과임.',SMALL),
    table(['사건 경보 진단','5초 경보','8초 경보'],[
        ['최초 경보 정밀도',pct(alarms['5s']['first_alert_precision'],2),pct(alarms['8s']['first_alert_precision'],2)],
        ['3초 이상 선행 사건 재현율',pct(alarms['5s']['first_alert_event_recall_3s'],2),pct(alarms['8s']['first_alert_event_recall_3s'],2)],
        ['오경보/드론·시간',f'{alarms["5s"]["false_alerts_per_drone_hour"]:.2f}',f'{alarms["8s"]["false_alerts_per_drone_hour"]:.2f}'],
        ['적중 사건 / 전체 대상 사건',f'{alarms["5s"]["event_hits_3s"]:,}/{alarms["5s"]["eligible_events"]:,}',f'{alarms["8s"]["event_hits_3s"]:,}/{alarms["8s"]["eligible_events"]:,}'],
        ['최초 경보 수',f'{alarms["5s"]["first_alerts"]:,}',f'{alarms["8s"]["first_alerts"]:,}'],
    ],[2.4,1.2,1.4]),
    note('높은 AUROC의 해석',f'AUROC {pred["risk_5s"]["auroc"]:.4f}는 위험 양성 표본과 음성 표본의 순위 구분 성능임. 단절의 {pred["risk_5s"]["auroc"]*100:.2f}%를 미리 탐지했다는 뜻이 아님. 실제 조기 사건 포착률과 오경보를 함께 확인해야 함.',AMBER),
    p(f'8초 규칙은 기존 조건(정밀도 80%·오경보 5회/h·경보 10건)을 충족하나 사건 재현율은 {pct(alarms["8s"]["first_alert_event_recall_3s"],3)}임. 전체 검증 선택 파일은 작성됨. enabled=True는 기존 코드의 수치 조건을 뜻하며, 새 개발 목표나 실기체 운용 승인을 충족했다는 뜻은 아님.',SMALL),
    refs(['L01','L03','C01','R01'])
])

alignment=D['alignment'];cover=alignment['coverage'];oracle=alignment['stored_label_oracle']
align_rows=[]
for key,label in [('eligible_events','전체 대상 사건'),('with_decision','3초 이상 선행 판단 시점 존재'),('with_support','그 시점의 공통 관측 지원'),('with_valid_current_up_label','현재 연결·유효 라벨 시점 존재'),('with_positive_stored_label','해당 head의 양성 라벨 시점 존재')]:
    align_rows.append([label,str(cover['5s']['common_support'][key]),str(cover['8s']['common_support'][key])])
align_rows.append(['저장 정답으로 가상 경보 시 적중',str(oracle['5s']['event_hits_3s']),str(oracle['8s']['event_hits_3s'])])
add('08. 시간 주기와 라벨 정렬 진단','용어 | 선행시간: 실제 단절 시작까지 남은 시간 / 라벨: 학습용 미래 정답',[
    table(['같은 기준 관측의 사건 진단','5초 경보','8초 경보'],align_rows,[3,1,1]),
    h('직접 확인한 사실'),
    p(f'5초 경보는 발생 3~5초 전에 경고해야 함. 현재 판단 주기는 3초임. 전체 1,779건 중 {1779-cover["5s"]["common_support"]["with_decision"]}건에는 이 구간 안의 판단 시점 자체가 없음. 8초 경보의 3~8초 선행 구간에는 모든 사건의 판단 시점이 존재함.'),
    p('관측 지원과 현재 연결 조건을 적용하면 기회가 추가로 감소함. 5초 경보의 onset_3to4s 학습 정답과 3~5초 선행 평가 구간도 완전히 같지 않음. 따라서 모델 용량뿐 아니라 시간 정의·판단 간격·관측 유효성을 함께 점검해야 함.'),
    note('가상 정답 경보의 한계','마지막 행은 미래 정답을 그대로 위험 점수로 사용한 오프라인 진단임. 실제 모델 성능이 아님. 모든 예측 알고리즘의 이론적 상한도 아님. 최초 경보와 해제 규칙의 영향까지 포함함.',AMBER),
    h('사건 수가 달라지는 이유'),
    p('본 페이지 1,779건은 고정 physics 8비행의 유향 링크 사건임. 초기 best 176의 8비행은 4,087건, 전체 96비행은 40,422건임. 정책과 평가 집합에 따라 실제 링크 사건 수·시점이 달라짐. 네트워크 전체 단절 횟수와도 구분함.'),
    p('이 진단은 어떤 주기가 더 높은 실제 예측 성능을 보장하는지 입증하지 않음. 1초 갱신의 효과는 동일 예측기·동일 사건·동일 관찰 시간으로 비교할 예정임.',SMALL),
    refs(['D16','D06','C01','C02'])
])

cal_rows=[]
for head,label in [('risk_5s','5초 위험'),('onset_3to4s','3~4초 시작'),('onset_3to7s','3~7초 시작')]:
    row=D['calibration']['joint176']['heads'][head]
    cal_rows.append([label,f'{row["before"]["brier"]:.6f}',f'{row["spatial_crossfit_after"]["brier"]:.6f}',signed(row['brier_change'],6)])
add('09. 확률 보정의 실제 효과','용어 | Brier: 확률과 실제 정답의 제곱 오차 평균. 낮을수록 양호',[
    p('위험 확률 80%를 출력한 사례들의 실제 발생률도 약 80%인지 확인하는 절차임. 보정은 기존 점수의 의미를 조절함. 관측에 없던 전조를 만들거나 탐지 순위 자체를 자동으로 개선하는 절차가 아님.'),
    table(['공동 PPO 176','Brier 보정 전','공간 교차 적합 후','후 - 전'],cal_rows,[1.4,1.4,1.6,1.2]),
    h('검증 방법'),
    p('공간 4개 fold 사용. 각 fold에서 6개 지역으로 양수 기울기의 logit 보정기를 적합하고 다른 2개 지역에서 평가함. 원래 발생 비율을 유지한 Bernoulli 손실 사용. 보정 전후의 동일 표본을 비교함.'),
    h('확인 결과'),
    p('176 모델의 Brier 개선은 확인되지 않음. 일부 ECE 수치가 작아져도 사건 탐지율이 개선됐다는 뜻은 아님. ECE는 같은 표본 수로 나눈 10개 구간의 평균 진단값이며 희귀 사건에서는 작은 숫자를 과대해석할 수 있음.'),
    p('출력 head를 독립 보정하면 포함 관계가 깨질 수 있음. 176 모델은 161,097개 검사 표본 중 원래 위반 0개, 보정 후 위반 15개임. 이 보정기를 운용에 적용하지 않음.'),
    note('해석 범위','과거 비교의 LightGBM은 5초 위험 Brier가 0.009680 → 0.008826으로 개선됨. Transformer에도 같은 효과가 있다는 근거는 없음. 이미 선택에 사용한 8개 맵의 교차 적합이므로 이전 모델 선택 편향도 남음.'),
    p('이번 보정 진단 뒤 새 경보 문턱이나 재배치 비행을 다시 평가한 것은 아님. 확률 보정 결과를 곧바로 경보 성능 향상으로 보고하지 않음.',SMALL),
    refs(['D18','D14','R02'])
])

add('10. 현재 부족한 부분과 근거','분류 | 확인 사실 / 원인 가설 / 미검증 과제를 구분함',[
    table(['항목','현재 근거','필요한 조치'],[
        ['조기 사건 포착',f'전체 96조건의 8초 사건 재현율 {pct(alarms["8s"]["first_alert_event_recall_3s"],3)}','정밀도·오경보와 함께 최소 재현율 기준 적용'],
        ['시간 주기·정답','5초 경보 사건 368건에 선행 판단 시점 없음','예측 갱신과 이동 주기 분리, 시간 라벨 재정의'],
        ['학습 조합 범위','166지역 포함. 지역×방향×간섭 조합 192/996','실패 조건 중심 보강. PPO 추가 경험은 별도 출처 관리'],
        ['표본 추출','현재 사전학습 추출은 risk_5s 양성 기준','시작 head·사건·어려운 정상 사례의 기여 비교'],
        ['학습 목적과 선택','PPO best는 제어 기준. 예측 보조계수 0.1','고정 예측기 + PPO와 공동학습의 통제 비교'],
        ['일반화 근거','8개 진단 지역 반복 사용, 전체 24지역 검증 완료','설정 고정 후 미사용 테스트 평가'],
        ['비용과 꼬리 지연','이동량 증가. 지연 변화는 집합별로 다름','지역별·전체 결과와 자원 비용을 함께 보고'],
        ['실제 환경','시뮬레이션 관측·RF 가정만 검증','실측 RF, 센서 오차, 바람·명령 지연 검증은 후속'],
    ],[1.1,2.2,2.3]),
    h('아직 원인으로 단정할 수 없는 항목'),
    p('공동 PPO가 예측을 훼손한다는 가설은 입증되지 않음. 공통 관측의 5초 AP는 사전학습 2800보다 PPO 176이 높음. 관측되지 않은 무작위 간섭은 예측에 한계를 줄 수 있으나, 현재 누락의 원인별 비율을 측정한 것은 아님.'),
    note('표본 수의 해석','고정 지도학습 자료의 5초 양성 표본은 61,253/4,123,030개임. 인접 시점은 상관되어 있음. 이를 독립 사건 수나 전체 PPO 경험 수로 해석하지 않음.'),
    refs(['L01','D14','D16','D17','C01'])
])

add('11. 착수한 예측 개선 실험','상태 | 200회 자료 준비 완료 / 학습·재생 상태는 아래 기준 시점에 고정',[
    table(['변경 항목','구현 내용'],[
        ['예측 대상','다음에 시작하는 1초 이상 지속 링크 단절'],
        ['시간 구간','[0,8)초를 0.5초 × 16개 + 8초 내 시작 없음. 총 17개 클래스'],
        ['표시용 구간','0~2초, 2~4초, 4~6초, 6~8초 확률로 합산 가능'],
        ['예측 입력 주기','기존 RF 관측으로 1초 시점의 과거 40프레임 재구성'],
        ['모델','현재 32차원·2층 encoder를 사전학습 2800에서 초기화. 시간 출력층 추가'],
        ['학습','예측 전용 6,000단계 목표. 기존 예측 head BCE 보조계수 0.25'],
        ['실행 자원','CPU 한 스레드, 낮은 우선순위. GPU 숨김. 본 학습 원본 보존'],
        ['기준 시점 진행',pilot_text+'. 내부 검증 점수는 최종 개선 입증과 구분'],
    ],[1.1,4.1]),
    h('오류 방지 검사'),
    p('7개 검사 통과. 모든 시간 구간 경계, 단발·지속 단절, 반복 사건의 첫 시작, 끝부분 정답 검열, 현재 단절 마스크, 미래 관측 차단, 확률 합산, 1초·3초의 동일 관찰 시간 분모를 검사함. 실제 encoder·출력층 gradient와 저장·복원 출력도 확인함. 실제 저장 비행의 순전파·역전파 검사 통과.'),
    note('평가 의미의 변경','새 정답은 다음 첫 단절임. 기존 head는 구간 안의 임의 단절을 포함할 수 있음. 저장 float16 관측으로 지도 입력도 재구성함. 새 AP를 기존 발표 AP와 단순 증감 비교하면 안 됨.',AMBER),
    p('이전 usable 버전도 1초 예측을 사용했음. 이번 작업은 현재 PPO의 3초 행동 주기와 예측 갱신을 분리하는 재도입·검증임. 새로 추가한 부분은 다음 첫 단절의 시간 구간 정답임.',SMALL),
    p('아직 미실행: 새 PPO 자료·조건 보강, 사건 식별자 단위 추출, 새 예측기의 경보 보정·실제 제어 연동. LightGBM 비교와 대형 Transformer 학습은 보류함.',SMALL),
    refs(['D19','D20','D21','C02'])
])

add('12. 다음 평가와 합격 판단 방식','용어 | 교차 적합: 맞춘 지역과 평가 지역을 나눠 반복하는 방법',[
    table(['순서','진행할 작업','판단 기준'],[
        ['1','시간 구간·관측 시각·검열 계약 확인','미래 입력 금지, 1초 지속 정의와 경계 일치'],
        ['2','같은 고정 예측기의 1초/3초 갱신 비교','동일 사건·관측·노출 시간, 사건 재현율·오경보 비교'],
        ['3','예측 전용 학습의 개선 전/후 비교','같은 정답 의미와 입력 재구성으로 기준 모델 재평가'],
        ['4','실패 조건·사건 중심 데이터 보강','새 조건·정책별 자료 출처와 공간 분리 기록'],
        ['5','확률 보정과 경보 문턱 검증','보정·선택 지역과 평가 지역 분리'],
        ['6','예측기 고정 + PPO 결합','제어 이득, 예측 품질, 이동·지연 비용 동시 확인'],
        ['7','전체 검증 후 설정 고정·최종 테스트','정책당 96조건 검증, 미사용 128조건 평가'],
    ],[.5,2.1,2.8]),
    h('제안한 개발 목표'),
    table(['항목','목표 예시','현재 해석'],[
        ['최초 경보 정밀도','90% 이상','경보 10건 이상 등 최소 표본 조건 함께 확인'],
        ['최소 3초 선행 사건 재현율','80% 이상','지원 불가·판단 유보 사건을 분모에서 몰래 제외하지 않음'],
        ['오경보','1회/드론·시간 이하','드론·링크·네트워크의 단위를 구분'],
        ['일반화와 확률 신뢰성','지역별 결과, 보정 도표·Brier','평균 한 개 수치만으로 통과 판정하지 않음'],
    ],[1.6,1.2,2.6]),
    p('위 목표는 요구사항을 구체화하기 위한 개발 제안임. 교수 지정값·달성값·운용 승인 기준이 아님. 관측 가능성과 대응 시간을 확인한 뒤 타당성을 검토해야 함.',SMALL),
    refs(['D15','D20','C01','C02','R01','R02'])
])

add('13. 용어 정리','보고서 전반에 동일한 정의와 단위를 적용함',[
    table(['용어','정의와 주의점'],[
        ['CTR','수색 3기가 통신 그래프에서 모두 연결된 시간의 비율. 링크 하나의 연결률과 다름.'],
        ['PDR','생성 패킷 중 실제 전달된 비율. 경보 정확도가 아님.'],
        ['지연 반영 PDR','전달 패킷에 min(1, 0.25초/전달 지연)을 가중한 비율.'],
        ['AP','정밀도-재현율의 요약 점수. 높을수록 양호. 정확도 %로 해석하지 않음.'],
        ['AUROC','임계값 전반의 순위 구분 성능. 조기 사건 탐지율과 다름.'],
        ['Precision / 정밀도','내보낸 양성 판단·경보 중 맞은 비율. 이 코드의 경보 정밀도에는 최소 3초 선행 조건이 직접 적용되지 않음.'],
        ['Recall / 재현율','정답 양성 중 포착한 비율. 표본 단위와 사건 단위의 분모를 구분함.'],
        ['선행시간','경보를 낸 시점부터 실제 단절 시작까지의 시간.'],
        ['Brier / ECE','Brier는 확률 제곱 오차. ECE는 확률과 발생률의 구간별 차이 요약. 낮은 값만으로 운용 신뢰를 보장하지 않음.'],
        ['p95','값의 95%가 이보다 낮은 분위수. 본 지연 표는 비행별 p95를 다시 평균한 값임.'],
        ['Checkpoint / best','저장한 모델 상태 / 지정 선택 기준에서 가장 좋은 저장본. 최신 저장본과 다를 수 있음.'],
        ['Episode / update','비행 한 회 / 비행 수집과 여러 minibatch 최적화를 포함한 PPO 갱신 묶음.'],
        ['지원 마스크','관측 이력·수신 정보가 입력 조건을 충족하는지 나타내는 값. 낮은 위험과 다른 의미.'],
        ['검열 / holdout','미래 정답을 끝까지 확인할 수 없는 사례 처리 / 개발 중 사용하지 않은 최종 평가 자료.'],
        ['pp / 상대 변화율','98% → 99%는 +1pp. 상대 변화율은 (새 값/기존 값 - 1) × 100%. AP 점수 차이와 구분함.'],
    ],[1.2,4.4]),
    refs(['C01','C02','R01','R02','R03'])
])

source_descriptions={
 'D01':'Drive 통합본 대조: 핵심 966개 파일, Python 161개 지문 일치.',
 'D03':'통합 ZIP 내부 FULL_PROJECT_MANIFEST.json: 두 전달본의 구성·출처.',
 'D05':'보고서 기준 시점의 완료 update·훈련 비행·최근 학습 통계.',
 'D06':'현재 best 176: 통신·예측·사건 경보 요약과 checkpoint 지문.',
 'D07':'통합본 verified_joint의 선택 update 1 평가.',
 'D08':'통합본 verified_fixed의 선택 update 1 평가.',
 'D09':'physics 기준선 8조건 평가.',
 'D10':'고정 지도학습 192회·검증 8회 경로와 원본 소스·자료 지문.',
 'D11':'사전학습 best 2800 선택 포인터.',
 'D13':'본 학습 정기 검증의 16~240 update 선택 이력.',
 'D14':'같은 관측·시점·정답에서 계산한 AP·AUROC·Brier.',
 'D16':'판단 시점·지원 범위·저장 라벨과 사건의 정렬 진단.',
 'D17':'고정 지도학습 자료 192회·166지역·양성 표본 분포.',
 'D18':'공간 교차 적합 확률 보정의 전후 수치·포함 관계 위반.',
 'D19':'새 예측 실험의 자료 준비 상태.',
 'D20':'새 1초 입력·시간 구간 정답·모델·학습 설정 계약.',
 'D21':'실제 관측의 순전파·역전파 검사 기록.',
 'D23':'통합본이 보존한 과거 실행량·비교 수치·검증 범위.',
}
source_short=[]
for sid,description in source_descriptions.items():
    entry=D['sources'][sid]
    path=entry.get('member') or Path(entry['path']).relative_to(ROOT).as_posix()
    source_short.append([sid,path,description])
add('14. 수치 근거와 재현 경로','모든 수치는 evidence/snapshot.json에 고정한 원자료에서 작성함',[
    table(['ID','파일 또는 ZIP 내부 경로','근거 내용'],source_short[:10],[.45,2.65,2.4]),
    h('원시 비행 기록 재계산'),
    p('[D06a]·[D07a]·[D08a]·[D09a]는 각 정책의 raw_paths가 가리키는 8개 JSON 메타데이터임. 각 파일의 SHA256을 스냅샷에 보존함. 평균 통신 지표를 다시 계산하여 summary와 절대오차 1e-12 이내 일치를 확인함. 지역·방향·간섭·seed 조합도 정책 사이에서 일치함.'),
    h('불확실성'),
    p('physics 대비 CTR 차이의 탐색적 지역 쌍 bootstrap 95% 구간: '+f'[{D["paired_bootstrap"]["search_connected_mean"]["lower"]*100:.3f}, {D["paired_bootstrap"]["search_connected_mean"]["upper"]*100:.3f}]pp. '+
      '8개 지역을 묶음 단위로 10,000회 재추출함. seed 271077. 이미 모델 선택에 쓴 검증 자료이므로 미사용 지역에 대한 독립 개선 증거로 확대하지 않음.',SMALL),
])

add('15. 추가 근거·검증 한계','코드 정의, 실행 기록, 일반 연구 배경을 분리함',[
    table(['ID','파일 경로','근거 내용'],source_short[10:],[.45,2.65,2.4]),
    p('<b>[C01] 원본 코드.</b> solo/model.py, data.py, targets.py, training.py, features.py, deploy.py, metrics.py, alarms.py 및 cooperative/world.py, continuity_world.py. 입력·라벨·손실·경보·통신 지표의 실제 구현 근거임.',SMALL),
    p('<b>[C02] 새 실험 코드.</b> prediction_improvement/core.py, run.py, test_core.py. 시간 구간 정답·인과 입력·학습·검사의 구현 근거임. 모든 원본 절대 경로와 SHA256은 스냅샷에 보존함.',SMALL),
    p('<b>[D24]</b> results/transformer/verification/cuda-check.json. 실제 Windows CUDA gradient·일치·재개 검사. <b>[D25]</b> training_ops/reports/runtime_configuration_comparison_20261029.json. 지문을 확인한 checkpoint의 실제 실행 설정.',SMALL),
    p('<b>기타 ID.</b> [D02] training_ops/reports/release_comparison_20261029.json: 최초 ZIP 대조. [D04] training_ops/status.json: 본 실행 명령. [D12] results/transformer/pretrain/latest.json: 누적 step. [D15] training_ops/prediction_validation_20261029_v2/results.json: 공통 관측·혼합·경보 진단. [D22] training_ops/machine.json: 장치·환경. 프로젝트 상대 경로의 기준은 drone_forecast_joint_training/임.',SOURCE),
    p('<b>[R01]</b> Saito &amp; Rehmsmeier (2015). Precision-Recall 평가 해석.<br/><link href="https://doi.org/10.1371/journal.pone.0118432" color="#087D86">https://doi.org/10.1371/journal.pone.0118432</link>',SMALL),
    p('<b>[R02]</b> Guo et al. (2017). On Calibration of Modern Neural Networks.<br/><link href="https://proceedings.mlr.press/v70/guo17a.html" color="#087D86">https://proceedings.mlr.press/v70/guo17a.html</link>',SMALL),
    p('<b>[R03]</b> Schulman et al. (2017). Proximal Policy Optimization Algorithms.<br/><link href="https://arxiv.org/abs/1707.06347" color="#087D86">https://arxiv.org/abs/1707.06347</link>',SMALL),
    p('연구 문헌은 평가 개념·방법의 일반 배경임. 본 UAV 환경에서 개선을 입증한 근거는 로컬 실행·평가 기록임. 실기체 성능, 새 개선 실험의 효과, 모든 조건의 높은 재현율은 아직 확인되지 않음.',SMALL),
])

detail_labels=[('risk_5s','5초 위험'),('onset_3to4s','3~4초 시작'),('onset_3to7s','3~7초 시작'),('onset_0to7s','0~7초 시작')]
add('16. 세부 예측 지표와 집계 규칙','best 176 전체 96조건 | 초기 8비행·고정 관측·과거 HTML 수치와 구분함',[
    table(['예측 출력','유효 표본 / 양성','AP','AUROC','Brier'],[
        [label,f'{pred[key]["samples"]:,} / {pred[key]["positives"]:,}',f'{pred[key]["average_precision"]:.4f}',f'{pred[key]["auroc"]:.4f}',f'{pred[key]["brier"]:.6f}']
        for key,label in detail_labels],[1.4,1.8,.8,.9,1.1]),
    Spacer(1,10),
    table(['문턱 0.5','정밀도','표본 재현율','F1','정확도'],[
        [label,pct(pred[key]['precision'],2) if pred[key]['tp']+pred[key]['fp'] else '정의 불가',pct(pred[key]['recall'],2),f'{pred[key]["f1"]:.4f}',pct(pred[key]['accuracy'],2)]
        for key,label in detail_labels],[1.4,1.1,1.3,.9,1.1]),
    Spacer(1,10),
    table(['혼동 행렬','TP','FP','FN','TN'],[
        [label]+[f'{pred[key][field]:,}' for field in ['tp','fp','fn','tn']] for key,label in detail_labels],[1.4,1.1,1.1,1.1,1.3]),
    h('용어와 해석'),
    p('TP: 맞게 잡은 양성. FP: 양성으로 잘못 판단한 음성. FN: 놓친 양성. TN: 맞게 판단한 음성. F1은 정밀도와 재현율의 조화 평균임. 네 수의 합이 유효 표본 수와 일치함.'),
    p(f'3~4초 시작 head는 문턱 0.5에서 양성 판단이 0건임. 정밀도 분모가 0이므로 정의 불가로 표시함. 이 경우에도 정확도는 {pct(pred["onset_3to4s"]["accuracy"],2)}로 높게 나옴. 희귀 사건에서 정확도만으로 예측 품질을 판단할 수 없는 사례임.'),
    p('표본 재현율은 중복된 인접 시점을 포함함. 사건 재현율은 지속 단절 시작 사건이 분모임. 경보 표의 horizon·최초 경보·해제 규칙도 적용되므로 두 재현율을 같은 값으로 해석하지 않음.',SMALL),
    refs(['L01','C01','R01'])
])


def insert_after(prefix,title,subtitle,story):
    index=next(i for i,(old_title,_,_) in enumerate(pages) if old_title.startswith(prefix))
    pages.insert(index+1,(title,subtitle,story))


legacy_intro=[
    note('기준 A: 이전 예측 전용 버전','이전 채팅 「캡스톤 디자인II」의 2026-10-25 20:54 배포 메시지와 라이브러리 업로드 성공 기록을 확인함. 스크린샷의 제목·설명·32배속 제어와 현재 보존된 HTML 템플릿이 일치함.'),
    Spacer(1,10),
    table(['찾은 파일','역할과 확인 범위'],[
        ['drone_forecast_usable_final.zip','이전 코드·학습 가중치·시연 묶음. 업로드 기록과 원본 링크 확인. 이번에는 재다운로드·재실행하지 않음.'],
        ['통신위험_전체임무_재생.html','스크린샷의 전체 임무 재생 화면. 예측기 연결, 예측 기반 재배치 미연결.'],
        ['Transformer_운용개선_결과.html','같은 배포 묶음의 성능표. PC 다운로드 폴더의 원본 HTML에서 수치 추출.'],
    ],[2,3.1]),
    h('세 기준의 구분'),
    table(['단계','핵심 내용'],[
        ['A. 이전 usable 버전','예측 전용 Transformer. 1초 예측. 위험 기반 재배치 연결 전.'],
        ['B. 2026-10-29 배포 v5','단독 Transformer + 공유 actor/중앙 critic PPO. 소규모 사전 검증.'],
        ['C. 현재 PC 본 학습','B와 같은 핵심 코드. 자료 확대·본 PPO 256회·정책당 96조건 검증 완료.'],
    ],[1.3,3.8]),
    p('앞선 보고서의 통합본 대비 개선은 B → C의 비교였음. 이번 요청에 따라 A의 HTML 수치를 추가함. A → C와 B → C의 비교를 혼합하지 않음.'),
    p('재생 HTML은 시연 자료임. 아래의 AP·경보 수치는 같은 묶음의 운용개선 결과 HTML에서 가져옴. 원문 표의 반올림 값을 사용하며 이전 가중치를 재평가하지 않음.',SMALL),
    refs(['U01','U02','U03','D01'])
]

legacy_compare=[
    table(['지표','이전 A의 HTML','현재 C의 전체 96조건','단순 수치 차이'],[
        ['5초 위험 AP',f'{legacy["ap"]:.3f}',f'{pred["risk_5s"]["average_precision"]:.4f}',f'약 {D["legacy_descriptive_gaps"]["ap"]:+.3f}'],
        ['5초 위험 Brier',f'{legacy["brier"]:.4f}',f'{pred["risk_5s"]["brier"]:.6f}',f'약 {D["legacy_descriptive_gaps"]["brier"]:+.4f}'],
        ['알림 정밀도',pct(legacy['high_precision'],1),pct(alarms['8s']['first_alert_precision'],2),f'{D["legacy_descriptive_gaps"]["precision_pp"]:+.1f}pp'],
        ['3초 선행 사건 재현율',pct(legacy['high_event_recall'],2),pct(alarms['8s']['first_alert_event_recall_3s'],3),f'약 {D["legacy_descriptive_gaps"]["event_recall_pp"]:+.2f}pp'],
        ['오경보/드론·시간',f'{legacy["high_false_alerts"]:.2f}',f'{alarms["8s"]["false_alerts_per_drone_hour"]:.4f}',f'약 {D["legacy_descriptive_gaps"]["false_alerts_per_drone_hour"]:+.2f}'],
    ],[1.55,1.15,1.6,1.2]),
    note('표의 차이는 성능 개선량이 아님','AP·Brier의 이전 값은 과거 48개 평가 비행, 현재 값은 96개 검증 비행임. 알림의 이전 값은 새 난수 8비행의 5초 위험 알림, 현재 값은 96비행의 8초 시작 경보임. 분모·정답·문턱·예측 주기가 다름. 이 산술 차이로 실제 향상이나 퇴행을 확정할 수 없음.',AMBER),
    h('이전 HTML 안에서 확인되는 개선'),
    p('동일 48비행의 기존 입력 → 선정 Transformer: AP 0.258 → 0.685(+0.427), Brier 0.0182 → 0.0109. 일반 경고 재현율 40.6% → 87.5%, 정밀도 13.1% → 29.3%. 이는 이전 연구 내부의 입력·학습 개선 결과임.'),
    p('이전 새 난수 8비행에서 보수 점수 규칙 → 고신뢰 알림: 정밀도 59.3% → 90.9%, 오경보 67.78 → 3.75회/h. 반면 3초 선행 첫 알림 재현율은 18.34% → 0.39%로 감소함. 높은 정밀도와 낮은 조기 포착의 상충은 이전에도 존재했음.'),
    h('현재 판단'),
    p('HTML의 예측 AP 숫자는 현재 전체 검증 값보다 높음. 현재 예측기가 이전보다 향상됐다는 동일 조건 증거는 없음. 현재의 확인된 발전은 예측-재배치-PPO 연결과 같은 조건에서의 통신 유지 개선임. 예측 자체의 성능 확보를 별도 과제로 유지함.'),
    refs(['U01','L01','L02','D14'])
]

structure_compare=[
    table(['항목','A. 이전 usable','C. 현재 v5'],[
        ['예측 구조','hidden 64, encoder 2층','hidden 32, encoder 2층'],
        ['예측기 파라미터',f'{D["model_structure"]["legacy_code_structure_parameters"]:,}',f'{D["model_structure"]["current_forecaster_parameters"]:,}'],
        ['과거 이력','20초, 0.5초 관측 40개','20초, 0.5초 관측 40개'],
        ['입력 특징','비콘·통신과 지도 선행 정보 12항목 보강','통신·마스크·의도와 지도·물리 46차원'],
        ['예측 주기','1초','현재 제어 연결에서는 3초'],
        ['정답·목적','5초·10초 위험 + 미래 최소 SINR 보조','5초 위험 + 3~4·3~7·0~7초 시작. 공동 PPO'],
        ['자료 구성','45지역 × 2방향 × 3조건 = 270비행','고정 지도학습 192비행, 166지역 + 추가 PPO 경험'],
        ['평가 구성','8개 평가 지역 48비행. 별도 새 난수 8비행','24개 검증 지역 96조건. 최종 테스트 미사용'],
    ],[1.1,2,2.2]),
    h('이전의 1초 갱신과 현재 개선안'),
    p('1초 갱신은 이전 A에 이미 있었음. 이번 개선은 PPO의 이동 명령 3초 주기와 예측 갱신을 분리하여 1초 예측을 재도입·검증하는 작업임. 다음 첫 단절을 0.5초 구간으로 나누는 정답은 새 실험의 변경 사항임.'),
    p('현재 모델이 과거보다 큰 Transformer인 것은 아님. 크기·정규화·지도 특징·정답·학습 목적이 함께 달라져 있음. 구조 차이만으로 AP 차이의 원인을 확정하지 않음.'),
    p('이전 파라미터 수는 HTML에 기재된 값과 보존된 구조 정의의 계산이 일치함. 현재 값은 예측기만 계산함. PPO actor·critic을 합산한 전체 시스템 크기와 비교하지 않음.',SMALL),
    refs(['U01','U03','C01','C02'])
]

full_rows=[]
for label,field,mult,unit in [('CTR','search_connected_mean',100,'pp'),('PDR','pdr',100,'pp'),('지연 반영 PDR','timely_pdr',100,'pp'),
                             ('평균 단절 시간(s)','outage_total_s',1,'s'),('비행별 최장 단절 평균(s)','outage_max_s',1,'s'),
                             ('평균 패킷 지연(ms)','delay_mean_s',1000,'ms'),('비행별 지연 p95 평균(ms)','delay_p95_s',1000,'ms'),
                             ('중계 이동(km)','relay_travel_m',.001,'km')]:
    prior=full_base['means'][field]*mult;current=full['means'][field]*mult
    full_rows.append([label,f'{prior:.3f}',f'{current:.3f}',f'{current-prior:+.3f} {unit}'])
full_control=[
    table(['같은 96조건','physics','best 176','차이'],full_rows,[2,1.1,1.1,1.4]),
    p('CTR·PDR 값의 단위는 %, 차이는 pp. 나머지 단위는 지표명에 표기함. 24지역·동일 방향·간섭·seed의 96쌍을 대조함. 각 비행의 동일 가중 평균임.',SMALL),
    h('전체 검증 결과'),
    p(f'CTR {signed((full["means"]["search_connected_mean"]-full_base["means"]["search_connected_mean"])*100)}pp, PDR {signed((full["means"]["pdr"]-full_base["means"]["pdr"])*100)}pp. 평균 단절 {(1-full["means"]["outage_total_s"]/full_base["means"]["outage_total_s"])*100:.1f}% 감소. 중계 이동은 {(full["means"]["relay_travel_m"]/full_base["means"]["relay_travel_m"]-1)*100:.1f}% 증가함.'),
    p(f'비행별 지연 p95 평균은 전체 96조건에서 {(1-full["means"]["delay_p95_s"]/full_base["means"]["delay_p95_s"])*100:.1f}% 감소함. 초기 8조건의 증가와 방향이 다름. 평가 범위 변경을 시간에 따른 학습 개선·악화로 해석하지 않음.'),
    note('남은 긴 단절',f'전체 최장 단절은 physics {full_base["summary"]["communication"]["longest_outage_s"]:.1f}초 → 현재 {full["summary"]["communication"]["longest_outage_s"]:.1f}초임. 초기 8조건의 17.5초는 전체 96조건의 최장값이 아님. 긴 간섭 조건의 실패가 남아 있음.',AMBER),
    p(f'96/96회 임무 완료·안전 감사 통과. 최소 분리 7.071m. 최대 비행시간 비율 {full["summary"]["communication"]["max_duration_ratio"]:.4f}. 최장 단절 사례: {text(full["worst_outage"]["region"])} / north_south / continuous.',SMALL),
    p('physics의 0확률 출력은 학습된 예측기가 아님. 그 AP를 새 예측기의 성능 향상 기준으로 사용하지 않음. 전체 검증은 개발·선택 자료이며 최종 미사용 테스트와 구분함.',SMALL),
    refs(['L01','L02','L03'])
]

pages.insert(1,('이전 기준 파일과 세 버전의 관계','원본 ZIP 재실행 없이 HTML에 기록된 수치를 사용함',legacy_intro))
pages.insert(2,('이전 HTML 수치와 현재 결과 비교','수치 병기와 성능 개선의 입증을 구분함',legacy_compare))
pages.insert(3,('이전 예측기와 현재 구조의 차이','모델 크기·입력·정답·판단 주기를 함께 기록함',structure_compare))
insert_after('05.','전체 96조건의 통신·재배치 성능','완료 검증 | 초기 8조건 집계와 분리',full_control)

links={entry['filename']:entry['library_url'] for entry in D['discovery']['files']}
extra_sources=[
    table(['ID','근거'],[
        ['U01','Downloads/Transformer_운용개선_결과.html. 이전 성능표 원문. 원본 사본·추출 표·SHA256 보존.'],
        ['U02','baseline_usable_20261025/discovery.json. 이전 채팅 배포 메시지와 라이브러리 업로드 성공 기록.'],
        ['U03','baseline_usable_20261025/model_structure.json. 예측기 구조·파라미터 수 대조. 이전 가중치 미재실행.'],
        ['L01','현재 best 176의 전체 96조건 summary 및 96개 원시 메타데이터.'],
        ['L02','physics의 전체 96조건 summary 및 같은 시나리오 96개 원시 메타데이터.'],
        ['L03','results/transformer/frozen_selection.json. 전체 검증 선택, best 176, test_used=False.'],
        ['L04~L06','PPO 256회 완료 기록, 파이프라인 종료 상태, 새 예측 실험의 진행 상태.'],
    ],[.8,4.4]),
    h('찾은 라이브러리 원본'),
    p('<link href="'+links['drone_forecast_usable_final.zip']+'" color="#087D86">drone_forecast_usable_final.zip - 원본 ZIP 열기</link>',SMALL),
    p('<link href="'+links['통신위험_전체임무_재생.html']+'" color="#087D86">통신위험_전체임무_재생.html - 스크린샷의 재생 파일</link>',SMALL),
    p('<link href="'+links['Transformer_운용개선_결과.html']+'" color="#087D86">Transformer_운용개선_결과.html - 수치의 원문 보고서</link>',SMALL),
    p('원본 링크는 같은 계정의 라이브러리 접근이 필요함. 이번 비교는 사용자 요청에 따라 PC에 저장된 HTML의 수치만 인용함. 모델을 재실행한 신규 동일 조건 비교 결과는 아님.',SMALL),
    h('스냅샷과 집계 규칙'),
    p('초기 8조건 수치는 이전 PDF의 스냅샷을 별도 보존함. 전체 96조건이 같은 summary 경로를 갱신하므로 역사적 8조건 값과 새 96조건 값을 혼합하지 않음. 새 전체 수치는 원시 메타데이터에서 재계산하고 scenario 키 96쌍의 일치를 확인함.'),
    p('예측 AP는 자료·발생 비율·시점·지원 마스크에 영향을 받음. 서로 다른 집단의 단순 차이와 같은 집단의 성능 변화는 별개임. 모든 경로·파일 지문·HTML 표·진행 상태는 interim_evidence/snapshot.json에 고정함.',SMALL),
]
insert_after('15.','이전 HTML·전체 검증의 추가 근거','기존 근거 목록에 새 기준과 최신 평가를 연결함',extra_sources)


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle('캡스톤 디자인 II 중간보고서')
    c.setAuthor('Capstone Design II / evidence-based local review')
    layout=[]
    for number,(title,subtitle,story) in enumerate(pages,1):
        if number>1:
            title=f'{number-1:02d}. '+__import__('re').sub(r'^\d+\.\s*','',title)
        c.setFillColor(NAVY);c.rect(0,PAGE_H-12,PAGE_W,12,fill=1,stroke=0)
        c.setFont('KR-Bold',19 if number==1 else 17.5)
        c.drawString(LEFT,PAGE_H-54,title)
        c.setFillColor(GRAY);c.setFont('KR',8.1);c.drawString(LEFT,PAGE_H-74,subtitle)
        remaining=list(story)
        frame=Frame(LEFT,61,WIDTH,PAGE_H-157,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,showBoundary=0)
        frame.addFromList(remaining,c)
        if remaining:
            raise RuntimeError(f'Page {number} overflow: {title}; remaining flowables={len(remaining)}')
        c.setStrokeColor(colors.HexColor('#DCE5EA'));c.line(LEFT,46,PAGE_W-LEFT,46)
        c.setFont('KR',7);c.setFillColor(GRAY)
        c.drawString(LEFT,32,'캡스톤 디자인 II 중간보고서 | '+CUTOFF)
        c.drawRightString(PAGE_W-LEFT,32,f'{number} / {len(pages)}')
        layout.append(dict(page=number,title=title))
        c.showPage()
    c.save()
    (OUT/'interim_evidence/page_index.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pdf=str(PDF),pages=len(pages),cutoff=CUTOFF,bytes=PDF.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    render()
