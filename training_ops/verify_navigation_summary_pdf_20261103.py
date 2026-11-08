"""Verify exact evidence numbers, text bounds, embedded fonts and page renders."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'output/pdf/캡스톤 디자인 II 중간보고서_예측개선_최종비교_20261103.pdf'
QA=ROOT/'output/pdf/evidence/navigation_summary_20261103.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(v):return '-' if v is None else f'{v*100:.2f}%'
def diff(v):return f'{v*100:+.2f}%p'


def main():
    evidence=read(QA)
    assert evidence['pdf_sha256']==sha(PDF)
    for p,value in evidence['source_sha256'].items():assert sha(ROOT/p)==value
    data=read(ROOT/'forecast_navigation/artifacts/final_results.json')
    tokens=[]
    for item in data['tasks'].values():
        for tag in (item['reference_tag'],item['ranking_champion']):
            m=item['models'][tag];raw=m['raw']
            tokens.extend(pct(raw[k]['ap']) for k in ('next5_on_common_currently_up_links','any3to7_on_common_support'))
            alarm=m['calibrated']['common_FA_budget1']
            if alarm:tokens.extend([pct(alarm['precision']),pct(alarm['event_recall_3s']),f'{alarm["false_or_late_per_drone_hour"]:.2f}',f'{alarm["event_hits_3s"]:,}/{alarm["eligible_directed_link_events"]:,}'])
        for block in (item['ranking_vs_reference'],item['contrasts'][0]):
            tokens.extend(diff(block[k]) for k in ('pooled_ap5_difference','pooled_ap37_difference'))
        interval=item['ranking_vs_reference']['uncertainty']['metrics']['macro_spatial_group_ap37_difference']['percentile95']
        tokens.extend(f'{v*100:+.2f}' for v in interval)
    svc=data['tasks']['service']
    for tag in (svc['reference_tag'],svc['ranking_champion']):
        a=svc['models'][tag]['calibrated']['alarms']
        tokens.extend([pct(a['precision']),pct(a['event_recall_3s']),f'{a["event_hits_3s"]}/{a["eligible_directed_link_events"]}'])
    prior=read(ROOT/'forecast_graph/artifacts/final_results.json')
    for item in prior['tasks'].values():
        a=item['models'][item['reference_tag']]['raw'];b=item['models'][item['selected_tag']]['raw']
        tokens.extend(diff(b[k]['ap']-a[k]['ap']) for k in ('next5_on_common_currently_up_links','any3to7_on_common_support'))
    texts=[];violations=[]
    with pdfplumber.open(PDF) as document:
        assert len(document.pages)==2
        for number,page in enumerate(document.pages,1):
            text=page.extract_text() or '';texts.append(text);assert len(text)>500
            for char in page.chars:
                if char.get('text','').strip() and (char['x0']<42 or char['x1']>page.width-42 or char['top']<25 or char['bottom']>page.height-22):
                    violations.append(dict(page=number,text=char['text'],bounds=[char['x0'],char['x1'],char['top'],char['bottom']]))
    assert not violations,violations[:10]
    normalized=re.sub(r'\s+',' ','\n'.join(texts))
    assert texts[0].splitlines()[0]=='캡스톤 디자인 II 중간보고서'
    missing=[v for v in tokens if v not in normalized];assert not missing,missing
    for value in ('94,080','6,000','42','90%','80%','31.56%','PPO','RF','최초 경보 판단 차이0건'):assert value in normalized,value
    embedded=[]
    for page in PdfReader(PDF).pages:
        for item in page['/Resources']['/Font'].values():
            font=item.get_object()
            if font.get('/FontDescriptor'):
                descriptor=font['/FontDescriptor'].get_object();embedded.append(bool(descriptor.get('/FontFile2') or descriptor.get('/FontFile3')))
    assert embedded and all(embedded)
    renders=sorted((ROOT/'output/pdf/evidence/navigation_summary_20261103_renders').glob('page-*.png'));assert len(renders)==2
    evidence.update(pages=2,exact_numeric_tokens_verified=len(tokens),title_verified=True,text_bounds_violations=0,
        embedded_fonts_verified=True,rendered_pages={str(p.relative_to(ROOT)):sha(p) for p in renders},visual_review_pending=True)
    QA.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':main()
