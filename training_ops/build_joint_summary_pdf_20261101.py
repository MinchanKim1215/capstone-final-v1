"""Four-page Korean capstone summary, derived from pinned local evidence."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from xml.sax.saxutils import escape
import hashlib
import json
import math

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Frame, Flowable
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/joint_summary_20261101'
PDF = OUT / '캡스톤 디자인 II 중간보고서_공동학습_결과및고찰_20261101.pdf'
RUN = ROOT / 'adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031'
COMPARISON = ROOT / 'training_ops/reports/expanded_results_20261101'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


PRED = read(COMPARISON / 'prediction.json')
COMM = read(COMPARISON / 'communication.json')
PROGRESS = read(RUN / 'progress.json')
BEST = read(RUN / 'best.json')
PROTOCOL = read(ROOT / 'adaptive_joint_expanded/artifacts/protocol.json')
B, A = PRED['models']['before'], PRED['models']['after']
CB, CA = COMM['before']['means'], COMM['after']['means']
AB, AA = B['alarms_1s'], A['alarms_1s']
NB, NA = B['next_onset_lt5s'], A['next_onset_lt5s']

pdfmetrics.registerFont(TTFont('KR', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KR-Bold', 'C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('KR', normal='KR', bold='KR-Bold', italic='KR', boldItalic='KR-Bold')
NAVY = colors.HexColor('#173247')
TEAL = colors.HexColor('#087D83')
RED = colors.HexColor('#AD493F')
GRAY = colors.HexColor('#596B78')
LIGHT = colors.HexColor('#F0F5F7')
LINE = colors.HexColor('#DCE5E9')
W, H = A4
LEFT = 46
WIDTH = W - 2 * LEFT

BODY = ParagraphStyle('body', fontName='KR', fontSize=9.4, leading=14.0,
                      textColor=NAVY, wordWrap='CJK', spaceAfter=6)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.1, leading=11.8, textColor=GRAY, spaceAfter=5)
HEADING = ParagraphStyle('h2', parent=BODY, fontName='KR-Bold', fontSize=11.8, leading=17,
                         spaceBefore=8, spaceAfter=7)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.8, leading=12.4, spaceAfter=0)
CELL_NUM = ParagraphStyle('cellnum', parent=CELL, alignment=2)
HEAD = ParagraphStyle('thead', parent=CELL, fontName='KR-Bold', textColor=colors.white)
SOURCE = ParagraphStyle('source', parent=SMALL, fontSize=7.4, leading=10.5, spaceAfter=4)


def clean(value):
    value = str(value)
    for dash in ('\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2212'):
        value = value.replace(dash, '-')
    return value


def p(text, style=BODY):
    return Paragraph(clean(text), style)


def h(text):
    return p(text, HEADING)


def table(headers, rows, widths, *, numeric=False, padding=6):
    widths = [WIDTH * width / sum(widths) for width in widths]
    items = [[p(escape(str(value)).replace('\n','<br/>'), HEAD) for value in headers]]
    for row in rows:
        cells = []
        for column, value in enumerate(row):
            text = escape(clean(value)).replace('\n','<br/>')
            cells.append(p(text, CELL_NUM if numeric and column else CELL))
        items.append(cells)
    result = Table(items, colWidths=widths, hAlign='LEFT', repeatRows=1)
    result.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('LINEBELOW',(0,0),(-1,0),.6,NAVY), ('LINEBELOW',(0,1),(-1,-1),.35,LINE),
        ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),padding), ('BOTTOMPADDING',(0,0),(-1,-1),padding),
    ]))
    return result


def note(title, text, color=TEAL):
    box = Table([[p('<b>'+escape(title)+'</b><br/>'+text, BODY)]], colWidths=[WIDTH])
    box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),LIGHT), ('LINEBEFORE',(0,0),(0,-1),3,color),
        ('LEFTPADDING',(0,0),(-1,-1),11), ('RIGHTPADDING',(0,0),(-1,-1),11),
        ('TOPPADDING',(0,0),(-1,-1),9), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return box


def pct(value):
    return f'{value*100:.2f}%'


def pp(before, after):
    return f'{100*(after-before):+.2f}%p'


def change(before, after):
    delta = 100*(after/before-1)
    return f'{abs(delta):.2f}% '+('증가' if delta > 0 else '감소')


def refs(text):
    return p(text, SOURCE)


class JointDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 106

    def draw(self):
        c = self.canv
        box_width = (WIDTH-36)/4
        centers = []
        labels = [
            ('공개 지도·출동 제약','대형 선택: 출발 전'),
            ('최근 20초 관측','예측 갱신: 1초'),
            ('단절 위험·공유 표현','PPO 재배치: 3초'),
            ('새 위치·통신 결과','다음 관측·학습 신호'),
        ]
        for i, (first, second) in enumerate(labels):
            x = i*(box_width+12)
            centers.append(x+box_width/2)
            c.setFillColor(LIGHT if i != 1 else colors.HexColor('#E1F0F0'))
            c.setStrokeColor(LINE)
            c.roundRect(x,43,box_width,50,5,fill=1,stroke=1)
            c.setFont('KR-Bold',8.2); c.setFillColor(NAVY)
            c.drawCentredString(x+box_width/2,74,first)
            c.setFont('KR',7.7); c.setFillColor(GRAY)
            c.drawCentredString(x+box_width/2,57,second)
            if i < 3:
                x0=x+box_width+2; x1=x+box_width+10
                c.setStrokeColor(TEAL); c.setLineWidth(1)
                c.line(x0,68,x1,68)
                path=c.beginPath(); path.moveTo(x1,68);path.lineTo(x1-3,70);path.lineTo(x1-3,66);path.close()
                c.setFillColor(TEAL);c.drawPath(path,fill=1,stroke=0)
        c.setStrokeColor(TEAL);c.setLineWidth(1.1)
        c.line(centers[3],43,centers[3],29)
        c.line(centers[3],29,centers[1],29)
        c.line(centers[1],29,centers[1],42)
        path=c.beginPath();path.moveTo(centers[1],43);path.lineTo(centers[1]-2.5,38);path.lineTo(centers[1]+2.5,38);path.close()
        c.setFillColor(TEAL);c.drawPath(path,fill=1,stroke=0)
        c.setFont('KR',7.7)
        c.drawCentredString((centers[1]+centers[3])/2,13,'학습 시: 예측 정답 오차 + 행동 보상으로 함께 갱신')


class OutageBars(Flowable):
    def __init__(self):
        super().__init__();self.width=WIDTH;self.height=121

    def draw(self):
        c=self.canv;x0=143;plot_width=WIDTH-193
        c.setFont('KR',7.5)
        for tick in (0,5,10,15,20,25):
            x=x0+tick/25*plot_width
            c.setStrokeColor(LINE);c.line(x,15,x,93)
            c.setFillColor(GRAY);c.drawCentredString(x,3,str(tick))
        groups=[('누적 단절: 비행 평균',CB['outage_total_s'],CA['outage_total_s'],83),
                ('전체 중 최장 단절',COMM['before']['global_longest_outage_s'],COMM['after']['global_longest_outage_s'],43)]
        for label,before,after,y in groups:
            c.setFillColor(NAVY);c.setFont('KR',8)
            c.drawRightString(x0-10,y-1,label)
            for value,dy,color in [(before,0,colors.HexColor('#93A6B4')),(after,-14,TEAL)]:
                c.setFillColor(color);c.rect(x0,y+dy,plot_width*value/25,9,fill=1,stroke=0)
                c.setFillColor(NAVY);c.setFont('KR',7.6)
                c.drawString(x0+plot_width*value/25+5,y+dy+1,f'{value:.2f}')
        c.setFont('KR',7.5);c.setFillColor(GRAY)
        c.drawString(0,106,'단절 시간 비교 | 공통 가로축 0~25초')
        c.setFillColor(colors.HexColor('#93A6B4'));c.rect(WIDTH-126,106,9,7,fill=1,stroke=0)
        c.setFillColor(GRAY);c.drawString(WIDTH-112,106,'확장 전')
        c.setFillColor(TEAL);c.rect(WIDTH-61,106,9,7,fill=1,stroke=0)
        c.setFillColor(GRAY);c.drawString(WIDTH-47,106,'확장 후')


pages=[]


def add(title, subtitle, story):
    pages.append((title,subtitle,story))


add('캡스톤 디자인 II 중간보고서', '공동학습 채택 배경·확장 학습 결과 및 고찰 | 기준일 2026-11-01', [
    note('핵심 판정', '예측 성능은 소폭 개선. 평균 누적 통신 단절은 <b>33.83% 감소</b>.<br/>3초 이상 선행 경보와 최악 단절은 미흡. 실사용 신뢰 수준 달성으로 해석할 수 없음.'),
    h('1. 공동학습 채택의 설계상 근거'),
    table(['문제','채택 이유'],[
        ['예측과 행동의 상호작용','중계 이동이 거리·차폐·다음 관측을 바꿈. 고정된 관측에서 잘 맞히는 예측이 임무 결과 개선으로 곧바로 이어지지는 않음.'],
        ['예측 정보의 행동 활용','예측 확률과 공유 표현을 재배치 정책에 연결. 행동 결과의 손실이 예측기까지 전달되어 임무에 유용한 표현을 학습하도록 구성.'],
        ['예측 성능의 별도 유지','실제 다음 단절 정답에 대한 예측 오차도 함께 학습. 통신 보상만으로 예측 확률이 정확해진다고 가정하지 않음.'],
    ],[1.1,3.8]),
    JointDiagram(),
    p('<b>공동학습의 의미:</b> Transformer 예측기와 PPO 재배치 정책을 연결하고, 예측 정답 손실과 정책 손실로 함께 갱신하는 방식. 예측기를 강화학습만으로 학습하는 구조는 아님.'),
    table(['학습 대상','실제 학습 신호'],[
        ['예측기·시간 출력층','정답 기반 시간 구간 분류 오차 + PPO 정책 손실의 기울기'],
        ['중계 이동 정책','연결 단절·패킷 품질·추가 이동·임무 시간 등을 반영한 PPO'],
        ['출발 전 대형 선택기','출동 제약을 만족하는 네 후보에서 선택. 임무 전체 결과로 PPO 갱신'],
    ],[1.25,3.65]),
    Spacer(1,7),
    p('네 후보: V형 8대(수색 3·중계 5), 횡렬 6대(4·2), 횡렬 10대(6·4), V형 6대(4·2). 괄호는 수색·중계 대수이며, 비행 중 대수는 바꾸지 않음.',SMALL),
    p('<b>연결 확인:</b> 최종 갱신 기록에서 PPO 정책 손실이 예측기 인코더와 시간 출력층에 전달된 값이 모두 0보다 큼. 실제 가중치 변화도 확인됨. 이는 공동학습 연결의 근거이며, 분리 학습 대비 우월성의 증명은 아님.',SMALL),
    p('1초는 예측 갱신 주기임. 미래 [0,8)초를 0.5초 간격 16개와 단절 없음으로 분류함. 미래 정답은 학습·사후 평가에만 사용하며 실행 입력에 제공하지 않음.',SMALL),
    refs('근거 [E3, E4, E5]. 운영 시에는 학습된 가중치로 판단하며, 위 갱신은 학습 과정에서 수행함.'),
])

prediction_rows=[
    ['5초 예측 AP',pct(NB['average_precision']),pct(NA['average_precision']),pp(NB['average_precision'],NA['average_precision'])],
    ['5초 예측 AUROC',pct(NB['auroc']),pct(NA['auroc']),pp(NB['auroc'],NA['auroc'])],
    ['5초 예측 정밀도',pct(NB['precision']),pct(NA['precision']),pp(NB['precision'],NA['precision'])],
    ['5초 예측 재현율',pct(NB['recall']),pct(NA['recall']),pp(NB['recall'],NA['recall'])],
    ['5초 예측 F1',pct(NB['f1']),pct(NA['f1']),pp(NB['f1'],NA['f1'])],
    ['Brier 점수: 낮을수록 좋음',f"{NB['brier']:.5f}",f"{NA['brier']:.5f}",change(NB['brier'],NA['brier'])],
    ['3초 이상 선행 사건 재현율',pct(AB['event_recall_3s']),pct(AA['event_recall_3s']),pp(AB['event_recall_3s'],AA['event_recall_3s'])],
    ['선행 최초 경보 정밀도',pct(AB['timely_first_alert_precision']),pct(AA['timely_first_alert_precision']),pp(AB['timely_first_alert_precision'],AA['timely_first_alert_precision'])],
    ['오경보·늦은 경보 / 드론·시간',f"{AB['false_or_late_per_drone_hour']:.2f}회",f"{AA['false_or_late_per_drone_hour']:.2f}회",'증가: 악화'],
]

add('2. 학습 구성과 예측 결과','비교 모델: 추가 학습 시작 v6 192회 → 선택된 후속 v7 64회',[
    table(['구분','확인 결과'],[
        ['후속 학습','10월 31일 16:25경 ~ 11월 1일 00:14. 약 7시간 49분. 정상 조기 종료.'],
        ['지도와 학습량','학습 지도 245개 전부 사용. 192회 갱신 × 8비행 = 1,536회 시뮬레이션 비행.'],
        ['새 지도 100개 배정','학습 79개·개발 검증 20개·최종 평가 지역 접경으로 제외 1개.'],
        ['초기화·모델 선택','v6 192회 전체 가중치 계승, 새 최적화 상태. 최고 기록 64회, 마지막 저장본 192회.'],
        ['종료 규칙','16회마다 정기 평가. 연결성 중심 최고 기록이 8회 연속 갱신되지 않아 종료. 예측 최우선 종료 규칙은 아님.'],
    ],[1,4.2]),
    h('동일 관측에서 예측기만 비교'),
    p('기존 4개·새 4개 개발 지역의 같은 <b>448,674개 관측</b>으로 비교. 관측 이력·관측 유무·이동 계획·건물/경로 특징·시각·정답을 동일하게 고정함. 정밀도·재현율과 경보는 확률 문턱 0.5 기준.',SMALL),
    p('5초 지표는 현재 연결된 링크의 다음 지속 단절 시작이 [0,5)초 안에 있는지를 평가함. 단절은 한쪽 방향이라도 이론적 홉 성공확률이 90% 미만인 상태가 1초 이상 지속되는 경우임. 실제 패킷 전달 실패와는 다른 정답임.',SMALL),
    table(['지표','확장 전','확장 후','변화'],prediction_rows,[2.15,1,1,1.2],numeric=True,padding=5.5),
    Spacer(1,9),
    note('결과 해석','5초 AP는 <b>1.20%p</b>, 재현율은 <b>3.64%p</b> 개선. 정밀도는 하락.<br/>3초 이상 선행 탐지는 3,801개 방향성 링크 사건 중 <b>103건 → 147건</b>. 오경보·늦은 경보는 53회 → 93회로 증가.'),
    Spacer(1,7),
    p('AP·AUROC는 위험 순위의 평가값임. AUROC 95.37%가 실제 단절의 95.37%를 미리 맞힌다는 뜻은 아님. %p는 두 백분율의 차이를 의미함.',SMALL),
    refs('근거 [E1, E4]. 후속 모델의 저장 확률 재현 최대 절대 차이 약 0.00000101. 32개 최종 평가 지역은 미사용.'),
])

comm_rows=[
    ['수색망 연결 유지율',pct(CB['search_connected_mean']),pct(CA['search_connected_mean']),pp(CB['search_connected_mean'],CA['search_connected_mean'])],
    ['패킷 전달률 PDR',pct(CB['pdr']),pct(CA['pdr']),pp(CB['pdr'],CA['pdr'])],
    ['기한 내 패킷 전달률',pct(CB['timely_pdr']),pct(CA['timely_pdr']),pp(CB['timely_pdr'],CA['timely_pdr'])],
    ['비행당 누적 단절 시간',f"{CB['outage_total_s']:.2f}초",f"{CA['outage_total_s']:.2f}초",change(CB['outage_total_s'],CA['outage_total_s'])],
    ['비행별 최장 단절의 평균',f"{CB['outage_max_s']:.2f}초",f"{CA['outage_max_s']:.2f}초",change(CB['outage_max_s'],CA['outage_max_s'])],
    ['전체 중 가장 긴 한 번의 단절',f"{COMM['before']['global_longest_outage_s']:.2f}초",f"{COMM['after']['global_longest_outage_s']:.2f}초",'25.93% 증가'],
    ['비행당 단절 횟수',f"{CB['outage_events']:.2f}회",f"{CA['outage_events']:.2f}회",change(CB['outage_events'],CA['outage_events'])],
    ['평균 패킷 지연',f"{CB['delay_mean_s']*1000:.2f}ms",f"{CA['delay_mean_s']*1000:.2f}ms",change(CB['delay_mean_s'],CA['delay_mean_s'])],
    ['비행별 95백분위 지연의 평균',f"{CB['delay_p95_s']*1000:.2f}ms",f"{CA['delay_p95_s']*1000:.2f}ms",change(CB['delay_p95_s'],CA['delay_p95_s'])],
    ['중계 이동 거리: 비행 평균',f"{CB['relay_travel_m']/1000:.2f}km",f"{CA['relay_travel_m']/1000:.2f}km",change(CB['relay_travel_m'],CA['relay_travel_m'])],
    ['평균 임무 시간',f"{CB['duration_s']:.2f}초",f"{CA['duration_s']:.2f}초",change(CB['duration_s'],CA['duration_s'])],
    ['임무 완료·안전 검사','8/8 통과','8/8 통과','유지'],
]

add('3. 통신·임무 결과','같은 8개 지도·방향·간섭 조건·난수 시드에서 각 정책으로 전체 임무 실행',[
    p('확장 전 모델의 임무 8개를 같은 조건으로 추가 실행해 비교함. 정책에 따라 재배치와 궤적이 달라지므로 예측기만의 효과와 구분해야 함. 시간·거리·지연은 비행 평균이며, ‘전체 중 가장 긴 단절’은 최댓값임.',SMALL),
    table(['지표','확장 전','확장 후','변화'],comm_rows,[2.15,1,1,1.2],numeric=True,padding=5.2),
    Spacer(1,12),
    OutageBars(),
    Spacer(1,6),
    note('성과와 비용','평균 단절·패킷 지연은 개선. 가장 길었던 단절은 <b>13.50초 → 17.00초</b>로 악화.<br/>중계 이동 거리는 2.73%, 임무 시간은 0.70% 증가. 평균 개선과 최악 상황 개선을 구분해야 함.',RED),
    Spacer(1,8),
    refs('근거 [E2]. 양쪽의 선택 대형은 이 8개 조건에서 모두 V형 8대. 대형 자동 선택 자체의 이득을 독립적으로 확인한 결과는 아님.'),
])

add('4. 고찰과 개선 방향','확인한 사실, 가능한 해석, 아직 검증하지 않은 효과를 구분',[
    h('고찰'),
    p('<b>공동학습 연결은 확인됨.</b> 예측기와 행동 정책의 실제 공동 갱신을 검증함. 다만 동일 학습량의 고정 예측기+PPO 또는 예측기 별도 학습과 비교하지 않았으므로, 이번 향상을 공동학습만의 효과로 귀속할 수 없음. [E3, E4]'),
    p('<b>통신 목표와 예측 목표의 균형 점검이 필요함.</b> 현재 모델 선택은 연결성 우선임. 평균 통신은 좋아졌지만 경보 오탐도 증가함. 예측 손실 계수 0.25는 예측 학습 비중 25%를 뜻하지 않으며, 두 손실이 상충하는지 별도 확인해야 함. [E3, E4]'),
    p('<b>정답과 경보가 다루는 범위가 다름.</b> 학습은 현재 연결된 링크의 첫 단절을 예측함. 사건 경보 평가는 회복 후 재단절도 포함함. 이 차이는 낮은 선행 재현율의 개선 대상이며, 유일한 원인으로 확정한 것은 아님. [E5]'),
    h('다음 개선 순서'),
    table(['순서','수행 내용'],[
        ['1. 목표 정렬','현재 단절·회복·재단절과 필요한 선행시간을 명시. 기존 첫 단절 지표와 전체 사건 경보 지표를 함께 유지.'],
        ['2. 입력 품질','이미 발행된 이동 계획·회전 예상·상대 정보 경과 시간 보완. 간섭 관측과 위치·지도 불확실성 추가 검토.'],
        ['3. 학습 방식 검증','같은 데이터·학습량에서 고정 예측기+PPO, 별도 학습, 공동학습 비교. 예측 지표와 오경보 제약을 모델 선택에 반영.'],
        ['4. 독립 확인','검증 자료에서 경보 문턱·확률 보정을 정한 뒤 독립 평가. 평균 단절과 최악 단절, 이동 비용을 함께 확인.'],
    ],[1,4.2],padding=5.5),
    Spacer(1,8),
    p('이번 학습에는 지도 확대와 기존 구조의 추가 학습을 반영함. 위 목표 정렬·추가 간섭 관측·자세 및 지도 오차 등 새 환경요소는 아직 반영하지 않음.',SMALL),
    note('해석의 범위','개발 검증 8개 지역에서 선택·비교한 시뮬레이션 결과임. 448,674개 시점은 서로 독립인 지역 448,674개를 뜻하지 않음. 지도 확대·추가 학습·정책 변화의 효과가 섞여 있음. 실제 기체는 미정이며 2.4GHz·고도 50m 가정. 최종 평가 32개 지역은 미사용.',RED),
    h('용어와 근거'),
    p('<b>PPO</b>: 행동 결과의 보상으로 정책을 갱신하는 강화학습 방법. <b>정밀도</b>: 발생시킨 양성 판단 중 맞은 비율. <b>재현율</b>: 실제 양성 중 탐지한 비율. <b>AP·AUROC</b>: 위험 순위 평가값. <b>Brier</b>: 확률과 정답의 제곱 오차, 낮을수록 좋음. <b>PDR</b>: 생성 패킷 중 전달된 비율.',SMALL),
    refs('[E1] 동일 관측 예측 재평가: 8비행, 448,674표본, 2026-11-01.<br/>[E2] 동일 시나리오 통신 재평가: 기존 4개·새 4개 지역, 2026-11-01.<br/>[E3] v7 모델·공동 손실·정책 입력 구현. [E4] 학습 종료·최고 모델·지도 기록.<br/>[E5] v7 시간 정답 생성·사건 경보 평가 구현.'),
    refs('PDF 첨부 evidence_20261101.json에 원자료 요약, 전체 파일 경로와 SHA-256을 보존함. 본 요약에는 과거 원본 176회 또는 통신 우수 96회 모델과의 직접 비교가 포함되지 않음.'),
])


def evidence():
    paths = [
        ('E1','동일 관측 예측 재평가',COMPARISON/'prediction.json'),
        ('E2','동일 시나리오 통신 재평가',COMPARISON/'communication.json'),
        ('E3','공동 손실 구현',ROOT/'adaptive_joint_expanded/learning.py'),
        ('E3','공유 예측기·정책 구현',ROOT/'adaptive_joint_expanded/model.py'),
        ('E3','통신 보상 구현',ROOT/'adaptive_joint_expanded/world.py'),
        ('E3','연결성 우선 기본 보상',ROOT/'drone_forecast_joint_training/cooperative/continuity_world.py'),
        ('E4','후속 학습 종료',RUN/'status.json'),
        ('E4','최고 모델 선택',RUN/'best.json'),
        ('E4','학습 갱신 기록',RUN/'progress.json'),
        ('E4','정기 평가·조기 종료 구현',ROOT/'adaptive_joint_expanded/driver.py'),
        ('E4','지도 목록',ROOT/'additional_maps_20261030/expanded_plan.json'),
        ('E4','후속 연결 및 종료',ROOT/'additional_maps_20261030/queue/status.json'),
        ('E4','대형 명칭: line_cross는 횡렬',ROOT/'drone_forecast_joint_training/swarm_sim/motion.py'),
        ('E5','정답 생성',ROOT/'adaptive_joint_expanded/targets.py'),
        ('E5','경보 평가',ROOT/'adaptive_joint_expanded/metrics.py'),
    ]
    files=[]
    for source_id,title,path in paths:
        payload=path.read_bytes()
        files.append(dict(source_id=source_id,title=title,path=str(path.resolve()),
                          sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
    return dict(created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
                report_title='캡스톤 디자인 II 중간보고서 - 공동학습 결과 및 고찰',
                files=files, prediction=PRED, communication=COMM,progress=PROGRESS,best=BEST,
                protocol_sha256=PRED['protocol_sha256'],scope='Development only; sealed final test unused',
                code_excerpts={name:(ROOT/path).read_text(encoding='utf-8') for name,path in [
                    ('joint_loss','adaptive_joint_expanded/learning.py'),
                    ('time_targets','adaptive_joint_expanded/targets.py'),
                    ('alarm_metrics','adaptive_joint_expanded/metrics.py')
                ]})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    TMP.mkdir(parents=True,exist_ok=True)
    draft=TMP/'report_body.pdf'
    c=canvas.Canvas(str(draft),pagesize=A4,pageCompression=1)
    c.setTitle('캡스톤 디자인 II 중간보고서 - 공동학습 결과 및 고찰')
    c.setAuthor('Capstone Design II')
    c.setSubject('Joint training rationale, matched evaluation results and discussion, 2026-11-01')
    for index,(title,subtitle,story) in enumerate(pages,1):
        c.setFillColor(NAVY);c.rect(0,H-9,W,9,fill=1,stroke=0)
        c.setFillColor(NAVY);c.setFont('KR-Bold',18.2)
        c.drawString(LEFT,H-50,title)
        c.setFillColor(GRAY);c.setFont('KR',8.6)
        c.drawString(LEFT,H-70,subtitle)
        c.setStrokeColor(LINE);c.line(LEFT,H-82,W-LEFT,H-82)
        remaining=list(story)
        frame=Frame(LEFT,61,WIDTH,H-158,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        frame.addFromList(remaining,c)
        if remaining:
            raise RuntimeError(f'Page {index} overflow: {len(remaining)} flowables remain')
        c.setStrokeColor(LINE);c.line(LEFT,45,W-LEFT,45)
        c.setFillColor(GRAY);c.setFont('KR',7.3)
        c.drawString(LEFT,30,'캡스톤 디자인 II | 공동학습 결과 및 고찰 | 2026-11-01')
        c.drawRightString(W-LEFT,30,f'{index} / {len(pages)}')
        c.showPage()
    c.save()
    data=evidence()
    payload=json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8')
    (TMP/'evidence_20261101.json').write_bytes(payload)
    reader=PdfReader(draft)
    writer=PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.add_attachment('evidence_20261101.json',payload)
    with PDF.open('wb') as stream:
        writer.write(stream)
    print(json.dumps(dict(pdf=str(PDF),pages=len(pages),bytes=PDF.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    main()
