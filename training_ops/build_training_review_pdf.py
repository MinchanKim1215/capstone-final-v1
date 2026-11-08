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
D=json.loads((OUT/'evidence/snapshot.json').read_text(encoding='utf-8'))
PDF=OUT/'캡스톤디자인II_학습평가_검토보고서_20261029.pdf'
pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KR-Bold','C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('KR',normal='KR',bold='KR-Bold',italic='KR',boldItalic='KR-Bold')
NAVY=colors.HexColor('#12324A');TEAL=colors.HexColor('#087D86')
GRAY=colors.HexColor('#5D6D7C');LIGHT=colors.HexColor('#EFF5F7')
AMBER=colors.HexColor('#95520F');RED=colors.HexColor('#A33D40')
PAGE_W,PAGE_H=A4
LEFT=49;WIDTH=PAGE_W-98
CUTOFF=D['as_of'][:16].replace('T',' ')+' KST'
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
common=D['common']['common'];pred=D['best_summary']['prediction'];alarms=D['best_summary']['alarms']
updates=D['progress']['committed_updates'];pilot=D['pilot_status'];cases=8

add('학습·평가 검토 보고서','캡스톤디자인 II | 중간보고 전체 코드 기준 | '+CUTOFF,[
    note('핵심 판정','통신 유지 성능 개선 확인. 동일 진단 관측에서 예측 순위 성능 개선 확인.<br/>사람이 신뢰할 조기 경보 기준은 미달. 새 예측 개선 실험의 효과는 아직 미측정.'),
    Spacer(1,14),
    table(['항목','기준 시점의 확인 결과'],[
        ['코드 기준','통합 ZIP과 핵심 파일 966개 일치. Python 소스 161개 일치. 같은 v5.'],
        ['본 학습',f'공동 PPO {updates}/256 update 완료. 훈련 비행 {D["progress"]["training_episodes"]:,}회. 제어 기준 best 176 유지.'],
        ['통합본 공동 best 대비',f'CTR {pct(old["search_connected_mean"])} → {pct(new["search_connected_mean"])}. 평균 단절 {old["outage_total_s"]:.3f} → {new["outage_total_s"]:.3f}초.'],
        ['동일 관측 예측',f'5초 위험 AP {common["pretrain1200"]["risk_5s"]["average_precision"]:.4f} → {common["joint176"]["risk_5s"]["average_precision"]:.4f}. 정확도 백분율이 아닌 점수.'],
        ['조기 경보 한계',f'best 176의 8초 경보: 정밀도 {pct(alarms["8s"]["first_alert_precision"],2)}, 3초 선행 사건 재현율 {pct(alarms["8s"]["first_alert_event_recall_3s"],2)}.'],
        ['현재 개선 실험',f'시간 구간 예측·1초 갱신 입력 준비. {pilot["complete"]}/200회 자료 처리 완료. 새 성능 결과 없음.'],
    ],[1.1,3.8]),
    h('읽기 기준'),
    p('① 통신 지표: 같은 8개 지역·시나리오에서 정책별 비행 결과 비교. ② 예측 지표: 같은 링크·관측·정답을 고정한 별도 재생 비교. ③ best 176 자체 비행: 현 모델의 동작 진단. 세 평가 집합을 혼합하지 않음.'),
    p('수치는 위 기준 시점에 고정함. 이후 진행되는 학습 상태는 본 PDF에 자동 반영되지 않음. 모든 결과는 시뮬레이션 내부 검증임.',SMALL),
    refs(['D01','D05','D06','D07','D14','D19'])
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
        ['본 학습 정기 검증','소규모 검증 기록','16~240 update의 15개 평가 시점'],
        ['제어 best','verified_joint 1 / verified_fixed 1','transformer_main 176'],
    ],[1.2,1.7,2.5]),
    p('epoch는 수집 표본을 학습하는 반복, minibatch는 한 번의 최적화에 쓰는 표본 묶음임. 본 update의 8회 비행과 최대 4 epoch는 서로 다른 단위임.',SMALL),
    Spacer(1,8),
    LineChart([(row['update'],row['score'][0]) for row in D['validation_history']],base['search_connected_mean']),
    p('점선: physics 기준선. 실선: 동일 8조건의 정기 검증 CTR. 최신 update가 항상 best는 아님. best는 안전·완료·비행시간 조건을 통과한 뒤 CTR, PDR, 최장 단절 순서로 선택함.',SMALL),
    p('본 학습은 verified_joint의 update 2에서 단순 재개한 실행이 아님. 사전학습 best와 기존 제어 기반 모델로 새 transformer_main을 시작함. 목표는 256 update이며 전체 검증은 정책당 96조건임.',SMALL),
    p('본 실행 시작: 2026-10-29 04:35 KST. 기준 시점까지 약 '+str(int((datetime.fromisoformat(D['as_of'])-datetime.fromisoformat(D['main_status']['started_at'])).total_seconds()//3600))+'시간 '+str(int((datetime.fromisoformat(D['as_of'])-datetime.fromisoformat(D['main_status']['started_at'])).total_seconds()%3600//60))+'분. 자료 수집·사전학습·PPO·검증을 포함한 경과 시간임. GPU만 계속 계산한 시간은 아님.',SMALL),
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
    note('평가 범위','현재 best 비교는 8개 지역의 축소 검증임. 전체 검증은 24지역 × 4조건 = 정책당 96조건. 최종 미사용 테스트는 32지역 × 4조건 = 정책당 128조건이며 본 보고서 작성 과정에서 사용하지 않음.'),
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
add('05. 통신·재배치 성능 변화','용어 | CTR: 수색 드론 전체 연결 시간 비율 / PDR: 생성 패킷 중 전달 비율',[
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
    note('함께 보고할 악화 지표',f'physics 대비 지연 p95의 평균 {(new["delay_p95_s"]/base["delay_p95_s"]-1)*100:.1f}% 증가. 중계 이동 {(new["relay_travel_m"]/base["relay_travel_m"]-1)*100:.1f}% 증가. 단절 지속시간 감소가 단절 횟수·꼬리 지연·이동 비용의 동시 개선을 뜻하지 않음.',AMBER),
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

add('07. best 176의 실제 경보 한계','용어 | 표본 재현율: 위험 시점 포착 / 사건 재현율: 단절 사건을 미리 포착',[
    table(['best 176 자체 비행','5초 위험 출력','3~7초 시작 출력'],[
        ['AP',f'{pred["risk_5s"]["average_precision"]:.4f}',f'{pred["onset_3to7s"]["average_precision"]:.4f}'],
        ['AUROC',f'{pred["risk_5s"]["auroc"]:.4f}',f'{pred["onset_3to7s"]["auroc"]:.4f}'],
        ['문턱 0.5에서 정밀도',pct(pred['risk_5s']['precision'],2),pct(pred['onset_3to7s']['precision'],2)],
        ['문턱 0.5에서 표본 재현율',pct(pred['risk_5s']['recall'],2),pct(pred['onset_3to7s']['recall'],2)],
    ],[2.4,1.2,1.4]),
    p('사건은 5개 중계기의 상대 기체별 유향 링크에서 집계함. 오경보 분모는 중계기들의 총 관찰 시간(드론·시간)임. 네트워크 전체 단절 횟수와 구분함.',SMALL),
    p('5초 위험 표본: 151,514개, 양성 4,824개(3.184%). 앞 페이지 공통 관측의 표본·발생률과 다름. AP 0.5081을 앞 페이지 AP 0.3835와 직접 증감 비교하지 않음.',SMALL),
    table(['사건 경보 진단','5초 경보','8초 경보'],[
        ['최초 경보 정밀도',pct(alarms['5s']['first_alert_precision'],2),pct(alarms['8s']['first_alert_precision'],2)],
        ['3초 이상 선행 사건 재현율',pct(alarms['5s']['first_alert_event_recall_3s'],2),pct(alarms['8s']['first_alert_event_recall_3s'],2)],
        ['오경보/드론·시간',f'{alarms["5s"]["false_alerts_per_drone_hour"]:.2f}',f'{alarms["8s"]["false_alerts_per_drone_hour"]:.2f}'],
        ['적중 사건 / 전체 대상 사건',f'{alarms["5s"]["event_hits_3s"]:,}/{alarms["5s"]["eligible_events"]:,}',f'{alarms["8s"]["event_hits_3s"]:,}/{alarms["8s"]["eligible_events"]:,}'],
        ['최초 경보 수',f'{alarms["5s"]["first_alerts"]:,}',f'{alarms["8s"]["first_alerts"]:,}'],
    ],[2.4,1.2,1.4]),
    note('높은 AUROC의 해석','AUROC 0.9415는 임의의 위험 양성 표본이 음성 표본보다 높은 점수를 받을 순위 성능을 뜻함. 단절의 94.15%를 미리 탐지했다는 뜻이 아님. 실제 조기 사건 포착률과 오경보를 함께 확인해야 함.',AMBER),
    p('8초 규칙은 과거 수치 조건(정밀도 80%·오경보 5회/h·경보 10건)을 충족하나 사건 재현율은 0.42%임. 최소 재현율 조건이 없었던 선택 기준의 한계임. JSON의 enabled=True는 전체 96조건 검증을 요구하는 운용 경보 승인과 다름.',SMALL),
    refs(['D06','C01','R01'])
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
    p('본 페이지 1,779건은 고정 physics 비행의 유향 링크 사건임. 앞 페이지 4,087건은 best 176 정책 비행의 유향 링크 사건임. 재배치가 달라지면 실제 링크 단절의 수·시점도 달라짐. 네트워크 전체 단절 횟수와도 다른 지표임.'),
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
        ['조기 사건 포착','176의 8초 사건 재현율 0.42%','정밀도·오경보와 함께 최소 재현율 기준 적용'],
        ['시간 주기·정답','5초 경보 사건 368건에 선행 판단 시점 없음','예측 갱신과 이동 주기 분리, 시간 라벨 재정의'],
        ['학습 조합 범위','166지역 포함. 지역×방향×간섭 조합 192/996','실패 조건 중심 보강. PPO 추가 경험은 별도 출처 관리'],
        ['표본 추출','현재 사전학습 추출은 risk_5s 양성 기준','시작 head·사건·어려운 정상 사례의 기여 비교'],
        ['학습 목적과 선택','PPO best는 제어 기준. 예측 보조계수 0.1','고정 예측기 + PPO와 공동학습의 통제 비교'],
        ['일반화 근거','8개 진단 지역을 선택에 반복 사용','전체 검증 후 설정 고정, 미사용 테스트 평가'],
        ['비용과 꼬리 지연','이동·비행별 지연 p95 평균 증가','제어 이득과 자원 비용을 함께 보고'],
        ['실제 환경','시뮬레이션 관측·RF 가정만 검증','실측 RF, 센서 오차, 바람·명령 지연 검증은 후속'],
    ],[1.1,2.2,2.3]),
    h('아직 원인으로 단정할 수 없는 항목'),
    p('공동 PPO가 예측을 훼손한다는 가설은 입증되지 않음. 공통 관측의 5초 AP는 사전학습 2800보다 PPO 176이 높음. 관측되지 않은 무작위 간섭은 예측에 한계를 줄 수 있으나, 현재 누락의 원인별 비율을 측정한 것은 아님.'),
    note('표본 수의 해석','고정 지도학습 자료의 5초 양성 표본은 61,253/4,123,030개임. 인접 시점은 상관되어 있음. 이를 독립 사건 수나 전체 PPO 경험 수로 해석하지 않음.'),
    refs(['D06','D14','D16','D17','C01'])
])

add('11. 착수한 예측 개선 실험','상태 | 코드 구현·검사 통과 / 자료 준비 중 / 개선 효과 미측정',[
    table(['변경 항목','구현 내용'],[
        ['예측 대상','다음에 시작하는 1초 이상 지속 링크 단절'],
        ['시간 구간','[0,8)초를 0.5초 × 16개 + 8초 내 시작 없음. 총 17개 클래스'],
        ['표시용 구간','0~2초, 2~4초, 4~6초, 6~8초 확률로 합산 가능'],
        ['예측 입력 주기','기존 RF 관측으로 1초 시점의 과거 40프레임 재구성'],
        ['모델','현재 32차원·2층 encoder를 사전학습 2800에서 초기화. 시간 출력층 추가'],
        ['학습','예측 전용 6,000단계 목표. 기존 예측 head BCE 보조계수 0.25'],
        ['실행 자원','CPU 한 스레드, 낮은 우선순위. GPU 숨김. 기존 CUDA run 유지'],
        ['기준 시점 진행',f'입력 자료 {pilot["complete"]}/200회 준비 완료. 시범 학습 결과 없음'],
    ],[1.1,4.1]),
    h('오류 방지 검사'),
    p('7개 검사 통과. 모든 시간 구간 경계, 단발·지속 단절, 반복 사건의 첫 시작, 끝부분 정답 검열, 현재 단절 마스크, 미래 관측 차단, 확률 합산, 1초·3초의 동일 관찰 시간 분모를 검사함. 실제 encoder·출력층 gradient와 저장·복원 출력도 확인함. 실제 저장 비행의 순전파·역전파 검사 통과.'),
    note('평가 의미의 변경','새 정답은 다음 첫 단절임. 기존 head는 구간 안의 임의 단절을 포함할 수 있음. 저장 float16 관측으로 지도 입력도 재구성함. 새 AP를 기존 발표 AP와 단순 증감 비교하면 안 됨.',AMBER),
    p('아직 미실행: 새 PPO 자료·조건 보강, 사건 식별자 단위 추출, 확률·경보 규칙 재검증, 실제 제어 연동. LightGBM 비교와 대형 Transformer 학습은 보류함.',SMALL),
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
add('16. 세부 예측 지표와 집계 규칙','best 176 자체 정책 비행 8회 | 이 표를 공통 관측의 AP와 혼합하지 않음',[
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
    p('3~4초 시작 head는 문턱 0.5에서 양성 판단이 0건임. 정밀도의 분모가 0이므로 정의 불가로 표시함. 이 경우에도 정확도는 99.06%로 높게 나옴. 희귀 사건에서 정확도만으로 예측 품질을 판단할 수 없는 사례임.'),
    p('표본 재현율은 중복된 인접 시점을 포함함. 사건 재현율은 지속 단절 시작 사건이 분모임. 경보 표의 horizon·최초 경보·해제 규칙도 적용되므로 두 재현율을 같은 값으로 해석하지 않음.',SMALL),
    refs(['D06','C01','R01'])
])


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle('캡스톤디자인 II - 학습·평가 검토 보고서')
    c.setAuthor('Capstone Design II / evidence-based local review')
    layout=[]
    for number,(title,subtitle,story) in enumerate(pages,1):
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
        c.drawString(LEFT,32,'캡스톤디자인 II | 학습·평가 검토 | '+CUTOFF)
        c.drawRightString(PAGE_W-LEFT,32,f'{number} / {len(pages)}')
        layout.append(dict(page=number,title=title))
        c.showPage()
    c.save()
    (OUT/'evidence/page_index.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pdf=str(PDF),pages=len(pages),cutoff=CUTOFF,bytes=PDF.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    render()
