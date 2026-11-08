"""Two-page Korean report from completed and pinned V11/V12 local evidence."""
from pathlib import Path
import hashlib
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'forecast_navigation/artifacts'
OUT=ROOT/'output/pdf/캡스톤 디자인 II 중간보고서_예측개선_최종비교_20261103.pdf'
QA=ROOT/'output/pdf/evidence/navigation_summary_20261103.json'
W,H=A4;L=43;WIDTH=W-L*2
NAVY=colors.HexColor('#18364A');BLUE=colors.HexColor('#245bad');GRAY=colors.HexColor('#586D7A');LIGHT=colors.HexColor('#F0F4F7');LINE=colors.HexColor('#D8E1E7')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(v):return '-' if v is None else f'{v*100:.2f}%'
def delta(v):return f'{v*100:+.2f}%p'
def clean(text):
    for c in ('\u2010','\u2011','\u2012','\u2013','\u2014','\u2212'):text=text.replace(c,'-')
    return text


class Page:
    def __init__(self,c,number,subtitle):
        self.c=c;self.number=number;self.y=H-112
        c.setFillColor(NAVY);c.setFont('KR-Bold',18);c.drawString(L,H-51,'캡스톤 디자인 II 중간보고서')
        c.setFont('KR',10.5);c.setFillColor(GRAY);c.drawString(L,H-75,subtitle)
        c.setStrokeColor(BLUE);c.setLineWidth(2);c.line(L,H-91,W-L,H-91)

    def para(self,text,small=False,bold=False,gap=5):
        style=ParagraphStyle('p',fontName='KR-Bold' if bold else 'KR',fontSize=8.2 if small else 9.4,
            leading=11.8 if small else 14.0,textColor=GRAY if small else NAVY,wordWrap='CJK')
        p=Paragraph(clean(text),style);_,height=p.wrap(WIDTH,1000);self.y-=height
        if self.y<50:raise ValueError(f'Page{self.number} overflow at {text[:40]} y={self.y}')
        p.drawOn(self.c,L,self.y);self.y-=gap

    def heading(self,text):self.y-=5;self.para(text,bold=True,gap=7)

    def table(self,headers,rows,widths):
        style=ParagraphStyle('cell',fontName='KR',fontSize=8.5,leading=12.0,wordWrap='CJK',textColor=NAVY)
        head=ParagraphStyle('head',parent=style,fontName='KR-Bold',textColor=colors.white)
        data=[[Paragraph(escape(clean(str(v))).replace('\n','<br/>'),head) for v in headers]]
        data += [[Paragraph(escape(clean(str(v))).replace('\n','<br/>'),style) for v in row] for row in rows]
        t=Table(data,colWidths=[WIDTH*v/sum(widths) for v in widths])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,0),(-1,-1),.35,LINE),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        _,height=t.wrap(WIDTH,1000);self.y-=height
        if self.y<50:raise ValueError(f'Page{self.number} table overflow y={self.y}')
        t.drawOn(self.c,L,self.y);self.y-=8

    def footer(self):
        self.c.setStrokeColor(LINE);self.c.setLineWidth(.5);self.c.line(L,39,W-L,39)
        self.c.setFillColor(GRAY);self.c.setFont('KR',7.3)
        self.c.drawString(L,27,'2026.11.03 | 연구용 시뮬레이션 | 실제 장비 검증 전')
        self.c.drawRightString(W-L,27,f'{self.number} / 2');self.c.showPage()


def main():
    if read(ART/'status.json')['state']!='completed':raise ValueError('Report only after all required work completed')
    d=read(ART/'final_results.json');gate=read(ART/'gate_result.json');audit=read(ART/'audit.json');runtime=read(ART/'runtime_checks.json');v11=read(ROOT/'forecast_graph/artifacts/final_results.json')
    if not audit['passed'] or not runtime['passed']:raise ValueError('Unverified study')
    pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'));pdfmetrics.registerFont(TTFont('KR-Bold','C:/Windows/Fonts/malgunbd.ttf'))
    pdfmetrics.registerFontFamily('KR',normal='KR',bold='KR-Bold',italic='KR',boldItalic='KR-Bold')
    OUT.parent.mkdir(parents=True,exist_ok=True);c=canvas.Canvas(str(OUT),pagesize=A4)
    c.setTitle('캡스톤 디자인 II 중간보고서 - 예측개선 최종비교');c.setAuthor('캡스톤 디자인 II')
    page=Page(c,1,'예측구조 개선·학습 및 동일 조건 비교 | 결과 요약')
    page.para('<b>판단:</b> 위험 순위 성능과 신뢰할 수 있는 선행 경보를 구분한다. 개발 경보 목표를 충족하지 못한 결과를 실사용 성능 확보로 표시하지 않는다.')
    page.heading('1. 같은 새42비행의 예측 성능')
    rows=[]
    for task,title in [('link','물리 링크'),('service','패킷 서비스')]:
        item=d['tasks'][task];a=item['models'][item['reference_tag']]['raw'];b=item['models'][item['ranking_champion']]['raw'];contrast=item['ranking_vs_reference']
        rows.append([title,pct(a['next5_on_common_currently_up_links']['ap'])+' / '+pct(a['any3to7_on_common_support']['ap']),
            pct(b['next5_on_common_currently_up_links']['ap'])+' / '+pct(b['any3to7_on_common_support']['ap']),delta(contrast['pooled_ap5_difference'])+' / '+delta(contrast['pooled_ap37_difference'])])
    page.table(['과제','기존 강한 기준\n5초 / 3~7초 AP','개선 순위 후보\n5초 / 3~7초 AP','차이\n5초 / 3~7초'],rows,[.8,1.25,1.25,1.35])
    page.para('개선 순위 후보: 두 과제 모두 보완 경로 모델과 기존 모델의50% 혼합. 선택 자료에서 미리 선정했다. 물리 기준은V9 혼합 모델, 서비스 기준은V11 혼합 모델이다. AP는 위험 순위 지표이며 정답률이 아니다.',small=True)
    intervals=[]
    for task,title in [('link','물리'),('service','서비스')]:
        z=d['tasks'][task]['ranking_vs_reference']['uncertainty']['metrics']['macro_spatial_group_ap37_difference']
        intervals.append(f'{title} [{z["percentile95"][0]*100:+.2f}, {z["percentile95"][1]*100:+.2f}]%p')
    page.para('공간 그룹 평균3~7초 AP 차이의 짝지은 재표집95% 구간: '+' / '.join(intervals)+'. 전체 관측 통합AP 차이의 구간과 구분한다.',small=True)
    page.heading('2. 경보 성능: 보정 FA1 문턱을 그대로 적용')
    rows=[]
    for task,title in [('link','물리'),('service','서비스')]:
        item=d['tasks'][task]
        for tag,caption in [(item['reference_tag'],'기준'),(item['ranking_champion'],'개선')]:
            a=item['models'][tag]['calibrated']['common_FA_budget1']
            rows.append([title+' '+caption,pct(a['precision']) if a else '운영점 없음',pct(a['event_recall_3s']) if a else '-',
                f'{a["false_or_late_per_drone_hour"]:.2f}' if a else '-',f'{a["event_hits_3s"]:,}/{a["eligible_directed_link_events"]:,}' if a else '-'])
    page.table(['과제·후보','경보 정밀도','3초 선행\n사건 탐지율','오경보·늦은\n경보/h','탐지한 사건\n/ 전체 사건'],rows,[1,1,1.05,1,1.25])
    page.para('FA1은 보정 자료에서 오경보1회/h 이하·최소50개 경보를 만족한 문턱이다. 확인 비행에서는 예산을 초과할 수 있다. 운영점 없음은 해당 보정 조건을 만족하는 문턱이 없다는 뜻이다. 경보 정밀도와 사건 탐지율의 분모는 다르다.',small=True)
    svc=d['tasks']['service'];a=svc['models'][svc['reference_tag']]['calibrated']['alarms'];b=svc['models'][svc['ranking_champion']]['calibrated']['alarms']
    page.para(f'<b>서비스 기본 문턱:</b>3초 선행 탐지 {a["event_hits_3s"]}/{a["eligible_directed_link_events"]}건({pct(a["event_recall_3s"])}) → {b["event_hits_3s"]}/{b["eligible_directed_link_events"]}건({pct(b["event_recall_3s"])}). 경보 정밀도 {pct(a["precision"])} → {pct(b["precision"])}. 개선안 오경보·늦은 경보 {b["false_or_late_per_drone_hour"]:.2f}/h. 실사용 경보 목표 미달.',small=True)
    page.para('<b>개발 목표:</b> 정밀도90%·3초 선행 탐지율80%·오경보/늦은 경보1회/드론·시간 이하. 높은 정밀도 경보를 우선한 연구용 API의 물리 후보는 기존 대조 트리를 유지했다. 순위 우수 후보와 구분한다.',small=True)
    page.heading('3. 경로 보완의 직접 효과: 학습량·초기 가중치 동일')
    rows=[]
    for task,title in [('link','물리 링크'),('service','패킷 서비스')]:
        a=d['tasks'][task]['contrasts'][0];rows.append([title,delta(a['pooled_ap5_difference']),delta(a['pooled_ap37_difference'])])
    page.table(['원래 경로 → 보완 경로','5초 AP 차이','3~7초 AP 차이'],rows,[1.5,1,1])
    page.para(f'추가 학습 전 선택45비행의 발행 시점 기준3~7초 위치RMSE: {gate["original_rmse_m"]:.2f}m → {gate["mission_preview_rmse_m"]:.2f}m ({gate["relative_rmse_reduction"]*100:.2f}% 감소). 경로 오차 감소율을 예측 향상률로 사용하지 않았다.',small=True)
    page.footer()
    page=Page(c,2,'예측구조 개선·학습 및 동일 조건 비교 | 구성·검증·고찰')
    page.heading('4. 구성: 미래 정답을 입력하지 않는 예측')
    page.para('<b>현재 임무·로컬 관측 → 실제 받은128바이트 비컨 → 시간 그래프·물리 전파 계산 → 링크/서비스 위험 예측 → 확률 보정·경보 문턱</b>')
    page.para('공유 경로는 이미 발행한3초 이동과 그 이후 잠정 경로다. 보완안은 임무 제어기의 현재 경로·회피·회복 상태를 반영한다. 실제 미래PPO 명령·RF·간섭·패킷 결과는 입력하지 않는다. 이후 제어가 달라지면 경로도 틀릴 수 있다.')
    page.table(['구분','세 가지 개선(V11)','경로 보완 대조(V12)'],[
        ['구조','경로 공유·시간 그래프·물리 잔차 보정','같은 그래프, 잠정 경로의 정보 보완'],
        ['신규 모델','그래프6개 + 트리 대조4개','원래/보완 경로 × 난수2개 = 4개'],
        ['훈련','245비행·94,080표본','동일 표본·정답·초기 가중치'],
        ['모델당 학습','그래프6,000업데이트','각각 추가6,000업데이트, 초기0회도 비교'],
        ['평가','선택45·보정45, 동결 후 새42비행','선택45·보정45, 별도 동결 후 새42비행'],
        ['PPO','기존 정책으로 고정','동일. 새 예측기는 지도학습'],
    ],[.9,1.75,1.75])
    prior=[]
    for task,title in [('link','물리'),('service','서비스')]:
        t=v11['tasks'][task];a=t['models'][t['reference_tag']]['raw'];b=t['models'][t['selected_tag']]['raw']
        prior.append(title+' '+delta(b['next5_on_common_currently_up_links']['ap']-a['next5_on_common_currently_up_links']['ap'])+' / '+delta(b['any3to7_on_common_support']['ap']-a['any3to7_on_common_support']['ap']))
    page.para('앞선V11 선정안의5초/3~7초 AP 변화: '+'; '.join(prior)+'. 각V11 내부의 같은42비행 비교이며V12 비행 수치와 직접 차감하지 않는다.',small=True)
    page.heading('5. 검증 결과')
    page.para(f'원래 의존 파일{audit["original_dependencies_preserved"]}개와V11 납품 파일{audit["parent_delivery_files_preserved"]}개를 보존했다. 비컨{audit["beacon_messages_roundtrip_checked"]:,}개에서 두 경로의 원래 패킷 복원을 검사했다. 실제 이동·RF·비컨 수신 성공·패킷 결과는 유지했다. 원래 최종32지도는 열지 않았다.')
    page.para(f'실제 내보낸 두API를{runtime["frames"]:,}프레임에서 실행했다. 두 경로의 입력이 일괄 평가와 정확히 일치했다. 최초 경보 판단 차이0건. 확률 최대 차이는 링크{runtime["max_abs_probability_error"]["link"]:.7g}, 서비스{runtime["max_abs_probability_error"]["service"]:.7g}다. CPU/GPU float32 허용오차0.0001 이내다. RF 결측은None, 신뢰 경보는 비활성이다.')
    page.heading('6. 고찰·용어')
    page.table(['항목','해석'],[
        ['AP / %p','AP는 위험 순위 성능. %p는 두 백분율의 절대 차이.'],
        ['물리 / 서비스','특정 무선 링크 단절 / 기한 내 패킷 전달률90% 미만의 지속 장애. 서로 다른 정답·분모.'],
        ['정밀도 / 탐지율','경보 중 맞은 비율 / 사건 발생3초 이상·7초 미만 전 최초 경보한 비율.'],
        ['성능 한계','위험 순위가 좋아져도 선행 사건 탐지가 부족하면 실사용 경보로 볼 수 없음. 오래 학습한 것만으로 신뢰성이 보장되지 않음.'],
        ['환경 한계','2.4GHz·50m·단순 전파·확률 채널·이상적TDMA. 실제 모듈·안테나·혼잡·분산 제어상태 동기화·실측 검증 미완료.'],
    ],[.9,3.7])
    page.para('새42비행은 알려진14지도의 새 난수 실현이다. 새로운 지형·실제 장비로 일반화하지 않는다. 모델·경보 문턱은 확인 전에 고정했다. 확인 결과에 맞춘 재선정과 자동PPO 교체는 하지 않았다.',small=True)
    page.para('근거: forecast_navigation/artifacts/final_results.json, freeze.json, audit.json, runtime_checks.json. 모든 후보·재표집 구간·세부 수치는 forecast_navigation/RESULTS_20261103.md 참조.',small=True)
    page.footer();c.save()
    QA.parent.mkdir(parents=True,exist_ok=True)
    value=dict(pdf=str(OUT),pdf_sha256=sha(OUT),expected_pages=2,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ART/'final_results.json',ART/'freeze.json',ART/'audit.json',ART/'runtime_checks.json',ART/'gate_result.json']},
        raw_numeric_tokens=[v for row in rows for v in row if '%' in str(v)],visual_review_pending=True)
    QA.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
