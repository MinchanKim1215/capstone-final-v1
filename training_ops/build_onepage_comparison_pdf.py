"""One-page summary of frozen, previously verified report evidence."""
from pathlib import Path
from xml.sax.saxutils import escape
import hashlib
import json

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
SOURCE = OUT / 'interim_evidence/snapshot.json'
PDF = OUT / '캡스톤_예측연결성_한장비교_20261029.pdf'
D = json.loads(SOURCE.read_text(encoding='utf-8'))
old = D['legacy_html']
current = D['latest']['joint']
base = D['latest']['physics']
risk = current['summary']['prediction']['risk_5s']
alarm = current['summary']['alarms']['8s']
common_old = D['common']['common']['pretrain1200']['risk_5s']
common_new = D['common']['common']['joint176']['risk_5s']

assert current['summary']['cases'] == base['summary']['cases'] == 96
assert common_old['samples'] == common_new['samples'] == 161341
assert old['samples'] == 4866832
assert alarm['event_hits_3s'] == 14 and alarm['eligible_events'] == 40422

pdfmetrics.registerFont(TTFont('KR', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KRB', 'C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('KR', normal='KR', bold='KRB', italic='KR', boldItalic='KRB')
NAVY = colors.HexColor('#133349')
TEAL = colors.HexColor('#087C83')
GRAY = colors.HexColor('#526575')
LIGHT = colors.HexColor('#F0F5F7')
LINE = colors.HexColor('#D9E3E8')
AMBER = colors.HexColor('#8D4B0B')
W, H = A4
M = 36
CW = W - 2 * M

CELL = ParagraphStyle('cell', fontName='KR', fontSize=10.0, leading=14.0,
                      textColor=NAVY, wordWrap='CJK')
HEAD = ParagraphStyle('head', parent=CELL, fontName='KRB', fontSize=9.5,
                      textColor=colors.white)
NUM = ParagraphStyle('num', parent=CELL, alignment=2)
SMALL = ParagraphStyle('small', parent=CELL, fontSize=8.3, leading=11.8,
                       textColor=GRAY)
BODY = ParagraphStyle('body', parent=CELL, fontSize=9.5, leading=14)

c = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1)
c.setTitle('캡스톤 디자인 II | 예측·연결성 한 장 비교')
c.setAuthor('Capstone evidence review')
c.setSubject('구버전 HTML·동일 관측 예측·동일 96조건 통신 지표의 구분 비교')
c.setFillColor(NAVY)
c.rect(0, H - 8, W, 8, fill=1, stroke=0)
y = H - M


def line(value, size=11, font='KR', color=NAVY, after=0):
    global y
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(M, y - size, value)
    y -= size * 1.38 + after


def para(value, style=SMALL, after=0):
    global y
    item = Paragraph(value, style)
    _, height = item.wrap(CW, H)
    item.drawOn(c, M, y - height)
    y -= height + after


def section(value, sub):
    global y
    line(value, 12, 'KRB', after=2)
    para(sub, after=6)


def table(headers, rows, widths, change_color=None):
    global y
    widths = [CW * v / sum(widths) for v in widths]
    data = [[Paragraph(escape(s), HEAD) for s in headers]]
    for row in rows:
        data.append([Paragraph(escape(s), CELL if i == 0 else NUM)
                     for i, s in enumerate(row)])
    item = Table(data, colWidths=widths, hAlign='LEFT')
    rules = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, 1), (-1, -1), .35, LINE),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    if change_color:
        for cell in data[1:]:
            cell[-1].style = ParagraphStyle('change', parent=NUM, textColor=change_color)
    item.setStyle(TableStyle(rules))
    _, height = item.wrap(CW, H)
    item.drawOn(c, M, y - height)
    y -= height


def pct(v, decimals=2):
    return f'{v * 100:.{decimals}f}%'


def pp(new, prior, decimals=2):
    return f'{(new - prior) * 100:+.{decimals}f}%p'


def relative(new, prior):
    return f'{(new / prior - 1) * 100:+.1f}%'


line('캡스톤 디자인 II 중간보고서', 15.5, 'KRB', after=2)
line('예측·연결성 성능 비교 | 한 장 요약', 20, 'KRB', after=4)
para('현재 모델: 본 학습 best 176 · 수치 기준: 2026-10-29 17:12 KST', after=12)

section('01  예측 성능: 구버전 HTML ↔ 현재',
        '서로 다른 평가 조건의 기록 비교. 아래 차이는 실제 향상·하락률로 확정할 수 없음.')
prediction_rows = [
    ['5초 위험 예측 AP*', pct(old['ap']), pct(risk['average_precision']),
     pp(risk['average_precision'], old['ap'])],
    ['위험 순위 구분 AUROC*', '기록 없음', pct(risk['auroc']), '-'],
    ['첫 경보 정밀도', pct(old['high_precision']), pct(alarm['first_alert_precision']),
     pp(alarm['first_alert_precision'], old['high_precision'])],
    ['3초 이상 조기 탐지율', pct(old['high_event_recall'], 3),
     pct(alarm['first_alert_event_recall_3s'], 3),
     pp(alarm['first_alert_event_recall_3s'], old['high_event_recall'], 3)],
    ['오경보 / 드론·시간', f"{old['high_false_alerts']:.3f}회",
     f"{alarm['false_alerts_per_drone_hour']:.3f}회",
     f"{alarm['false_alerts_per_drone_hour'] - old['high_false_alerts']:+.3f}회"],
]
table(['평가 항목', '구버전 HTML', '현재 176', '수치 차이'], prediction_rows, [40, 20, 20, 22])
y -= 6
para('구버전: AP 48비행 / 경보 새 난수 8비행. 현재: 96비행. 경보 대상도 5초 위험 → 8초 시작 경보로 다름. '
     '조기 탐지율은 전체 대상 단절 중 3초 이상 먼저 알린 사건 비율.', after=12)

section('02  연결성: 기존 제어 ↔ 현재',
        '동일 96조건 비교. 기존 제어는 physics 기준이며, 구버전 HTML의 연결률 기록은 없음.')
b = base['means']
n = current['means']
communication_rows = [
    ['연결 유지율 (CTR)', pct(b['search_connected_mean']), pct(n['search_connected_mean']),
     pp(n['search_connected_mean'], b['search_connected_mean'], 3)],
    ['패킷 전달률 (PDR)', pct(b['pdr']), pct(n['pdr']), pp(n['pdr'], b['pdr'], 3)],
    ['제시간 패킷 전달률', pct(b['timely_pdr']), pct(n['timely_pdr']),
     pp(n['timely_pdr'], b['timely_pdr'], 3)],
    ['비행당 총 단절시간 평균', f"{b['outage_total_s']:.2f}초", f"{n['outage_total_s']:.2f}초",
     relative(n['outage_total_s'], b['outage_total_s'])],
    ['전체 최장 단절', f"{base['worst_outage']['outage_max_s']:.1f}초", f"{current['worst_outage']['outage_max_s']:.1f}초",
     relative(current['worst_outage']['outage_max_s'], base['worst_outage']['outage_max_s'])],
    ['평균 패킷 지연', f"{b['delay_mean_s'] * 1000:.2f}ms", f"{n['delay_mean_s'] * 1000:.2f}ms",
     relative(n['delay_mean_s'], b['delay_mean_s'])],
    ['비행별 지연 p95 평균', f"{b['delay_p95_s'] * 1000:.2f}ms", f"{n['delay_p95_s'] * 1000:.2f}ms",
     relative(n['delay_p95_s'], b['delay_p95_s'])],
    ['중계 드론 총 이동거리 평균', f"{b['relay_travel_m'] / 1000:.2f}km", f"{n['relay_travel_m'] / 1000:.2f}km",
     relative(n['relay_travel_m'], b['relay_travel_m'])],
]
table(['평가 항목', '기존 제어', '현재 176', '증감'], communication_rows, [40, 20, 20, 22])
y -= 6
para('CTR: 수색 드론의 기지국 연결 비율 평균. p95: 지연이 큰 쪽 5%의 경계값. 시간·지연은 감소가 유리. '
     '중계 이동거리 증가는 추가 이동 비용.', after=12)

# Separately show the matched prediction comparison so a reader can distinguish
# a measured v5 improvement from the non-matched published legacy HTML values.
ap_change = pp(common_new['average_precision'], common_old['average_precision'])
auc_change = pp(common_new['auroc'], common_old['auroc'])
box_text = ('<b>동일 관측에서 확인한 예측 변화: 배포 v5 사전학습 → 현재 176</b><br/>'
            f"AP* {pct(common_old['average_precision'])} → {pct(common_new['average_precision'])} "
            f"({ap_change}, 상대 증가 {relative(common_new['average_precision'], common_old['average_precision'])})<br/>"
            f"AUROC* {pct(common_old['auroc'])} → {pct(common_new['auroc'])} ({auc_change}). "
            '고정 8관측 결과. 최종 독립 시험 아님.')
box = Table([[Paragraph(box_text, BODY)]], colWidths=[CW])
box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), LIGHT),
                         ('LINEBEFORE', (0, 0), (0, -1), 3, TEAL),
                         ('LEFTPADDING', (0, 0), (-1, -1), 11),
                         ('RIGHTPADDING', (0, 0), (-1, -1), 9),
                         ('TOPPADDING', (0, 0), (-1, -1), 8),
                         ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
_, height = box.wrap(CW, H)
box.drawOn(c, M, y - height)
y -= height + 9

para('*AP·AUROC는 0~1 점수를 100배로 표시함. 실제 적중률과 다름. %p는 퍼센트포인트. '
     '예: 97% → 98%는 +1%p.', after=3)
para('근거: 통합 중간보고서 3·10·11·12쪽 및 보존 스냅샷(interim_evidence/snapshot.json). '
     '본 표는 시뮬레이션 검증 결과이며 새 1초 예측 실험 결과는 제외함.')
assert y >= 27, f'One-page content overflow: bottom={y:.2f}'
c.save()

reader = PdfReader(PDF)
assert len(reader.pages) == 1
value = reader.pages[0].extract_text()
for token in ['68.50%', '47.89%', '90.90%', '80.00%', '0.390%', '0.035%',
              '97.67%', '98.36%', '-29.1%', '29.61%', '38.35%', '94.46%', '기록 없음']:
    assert token in value, token
violations = []
with pdfplumber.open(PDF) as document:
    for char in document.pages[0].chars:
        if char['text'].strip() and (char['x0'] < M - 1 or char['x1'] > W - M + 1 or char['bottom'] > H - 24):
            violations.append(char['text'])
assert not violations, violations[:10]
qa = dict(pdf=str(PDF), pages=1, bottom_y=y,
          source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
          data_cutoff=D['as_of'], predictions=prediction_rows, communications=communication_rows,
          automated_checks='one page, numeric text, and text bounds passed', visual_review_pending=True)
qa_path = OUT / 'interim_evidence/onepage_qa.json'
qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(pdf=str(PDF), pages=1, bottom_y=y, qa=str(qa_path)), ensure_ascii=False))
