from pathlib import Path
import hashlib
import json
import re
import pdfplumber
from pypdf import PdfReader
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[1]
folder=root/'output/pdf'
pdf=folder/'캡스톤디자인II_학습평가_검토보고서_20261029.pdf'
data=json.loads((folder/'evidence/snapshot.json').read_text(encoding='utf-8'))
texts=[]
violations=[]
with pdfplumber.open(pdf) as book:
    for number,page in enumerate(book.pages,1):
        value=page.extract_text() or ''
        assert len(value)>150,(number,len(value))
        texts.append(value)
        for char in page.chars:
            if char.get('text','').strip() and (char['x0']<45 or char['x1']>page.width-44 or char['top']<24 or char['bottom']>page.height-23):
                violations.append(dict(page=number,text=char.get('text')))
assert len(texts)==17 and not violations
full='\n'.join(texts)
expected=['98.221','97.543','32.375','0.3835','0.2961','80.95','0.42',
          str(data['progress']['committed_updates'])+'/256',str(data['pilot_status']['complete'])+'/200',
          '16:33','151,514','145,957','0.009676','11시간 58분']
assert all(value in full for value in expected),[value for value in expected if value not in full]
used=set(re.findall(r'\[(?:D\d\d[a-z]?|C\d\d|R\d\d)\]',full))
defined=set(re.findall(r'\[(?:D\d\d[a-z]?|C\d\d|R\d\d)\]','\n'.join(texts[14:16])))
# Source-index table cells display the ID without square brackets.
for row in ['D01','D03','D05','D06','D07','D08','D09','D10','D11','D13','D14','D16','D17','D18','D19','D20','D21','D23']:
    assert row in '\n'.join(texts[14:16])
    defined.add('['+row+']')
assert not used-defined,used-defined
for row in data['best_summary']['prediction'].values():
    assert sum(row[key] for key in ['tp','fp','fn','tn'])==row['samples']
reader=PdfReader(pdf)
embedded=[]
for page in reader.pages:
    for reference in page['/Resources']['/Font'].values():
        font=reference.get_object()
        if font.get('/FontDescriptor'):
            descriptor=font['/FontDescriptor'].get_object()
            embedded.append(bool(descriptor.get('/FontFile2') or descriptor.get('/FontFile3')))
assert embedded and all(embedded)
(folder/'evidence/extracted_text.txt').write_text(full,encoding='utf-8')
images=sorted((root/'tmp/pdfs/training_review').glob('page-*.png'))
assert len(images)==17
w,h=278,404
sheet=Image.new('RGB',(w*4,h*5),(221,228,232))
draw=ImageDraw.Draw(sheet)
for number,path in enumerate(images):
    im=Image.open(path).convert('RGB')
    im.thumbnail((w-12,h-25))
    x=(number%4)*w+6;y=(number//4)*h+20
    sheet.paste(im,(x,y));draw.text((x,y-16),str(number+1),fill=(20,40,60))
sheet.save(root/'tmp/pdfs/training_review/contact.png')
result=dict(pages=17,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            expected_numeric_tokens_verified=len(expected),source_reference_ids_verified=len(used),
            text_bounds_violations=0,embedded_fonts_verified=True,confusion_matrix_totals_verified=True,
            all_page_renderings_created=True,visual_review_pending=True,cutoff=data['as_of'])
(folder/'evidence/qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
