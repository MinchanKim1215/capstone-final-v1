from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
import pdfplumber
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
PDF=OUT/'캡스톤 디자인 II 중간보고서_20261029.pdf'
d=json.loads((OUT/'interim_evidence/snapshot.json').read_text(encoding='utf-8'))
page_index=json.loads((OUT/'interim_evidence/page_index.json').read_text(encoding='utf-8'))
texts=[];violations=[]
with pdfplumber.open(PDF) as document:
    for number,page in enumerate(document.pages,1):
        value=page.extract_text() or ''
        texts.append(value)
        assert len(value)>150,(number,len(value))
        for char in page.chars:
            if char.get('text','').strip() and (char['x0']<45 or char['x1']>page.width-44 or char['top']<24 or char['bottom']>page.height-23):
                violations.append(dict(page=number,text=char['text'],bounds=[char['x0'],char['x1']]))
assert not violations,violations[:10]
assert texts[0].splitlines()[0]=='캡스톤 디자인 II 중간보고서',texts[0].splitlines()[:2]
assert len(texts)==22==len(page_index)
all_text='\n'.join(texts)
normalized_text=re.sub(r'\s+',' ',all_text)
required=['0.685','0.258','0.4789','98.357','97.668','106.0','108.0','256/256','2,048','40,422','14/40,422','0.035','81,923','22,244','3.75','90.9','0.39','라이브러리','평가 집합','실제 향상이나 퇴행을 확정할 수 없음']
assert all(token in normalized_text for token in required),[token for token in required if token not in normalized_text]
assert '수치 차이는 성능 개선량이 아님' not in all_text or '성능 개선량이 아님' in all_text
# A single checkpoint is evaluated on two different sets, so preserve both values.
assert '0.5081' in all_text and '0.4789' in all_text
for prefix in ['joint','physics']:
    assert d['latest'][prefix]['summary']['cases']==96
assert d['legacy_html']['samples']==4866832
for metric in d['latest']['joint']['summary']['prediction'].values():
    assert sum(metric[key] for key in ['tp','fp','fn','tn'])==metric['samples']
pdf=PdfReader(PDF)
embedded=[];urls=[]
for page in pdf.pages:
    for reference in page['/Resources']['/Font'].values():
        font=reference.get_object()
        if font.get('/FontDescriptor'):
            descriptor=font['/FontDescriptor'].get_object()
            embedded.append(bool(descriptor.get('/FontFile2') or descriptor.get('/FontFile3')))
    for reference in page.get('/Annots',[]):
        annotation=reference.get_object()
        if annotation.get('/A',{}).get('/URI'):
            urls.append(str(annotation['/A']['/URI']))
assert embedded and all(embedded)
for item in d['discovery']['files']:
    assert item['library_url'] in urls,item['filename']
images=sorted((ROOT/'tmp/pdfs/capstone_interim').glob('page-*.png'))
assert len(images)==22
w,h=278,404
contact=Image.new('RGB',(w*5,h*5),(224,230,234));draw=ImageDraw.Draw(contact)
for index,path in enumerate(images):
    image=Image.open(path).convert('RGB');image.thumbnail((w-12,h-25))
    x=(index%5)*w+6;y=(index//5)*h+20
    contact.paste(image,(x,y));draw.text((x,y-16),str(index+1),fill=(20,40,60))
contact.save(ROOT/'tmp/pdfs/capstone_interim/contact.png')
(OUT/'interim_evidence/extracted_text.txt').write_text(all_text,encoding='utf-8')
result=dict(pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),pages=22,title_verified=True,
            numeric_and_scope_tokens_verified=len(required),legacy_library_links_verified=3,
            text_bounds_violations=0,embedded_fonts_verified=True,confusion_matrix_totals_verified=True,
            source_comparison_scope='Published legacy HTML values only; no old-weight rerun',
            cutoff=d['as_of'],visual_review_pending=True)
(OUT/'interim_evidence/qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
print(json.dumps(page_index,ensure_ascii=False))
