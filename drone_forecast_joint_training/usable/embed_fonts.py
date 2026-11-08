"""HTML의 한국어 항목형 문구 검사와 자체 포함 글꼴 생성."""
import base64,json,re
from pathlib import Path
from lxml import html
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools import subset
from swarm_sim.regions import ROOT,write_json


def main():
    paths=[ROOT/'docs/Transformer_운용개선_결과.html',ROOT/'docs/통신위험_전체임무_재생.html']
    texts=[p.read_text() for p in paths];chars=set(''.join(texts));assets=ROOT/'docs/assets'
    source=assets/'NotoSansKR.ttf';encoded={}
    for weight,label in ((400,'Regular'),(700,'Bold')):
        font=TTFont(source);instantiateVariableFont(font,{'wght':weight},inplace=True)
        options=subset.Options();options.flavor='woff';sub=subset.Subsetter(options=options);sub.populate(unicodes={ord(c) for c in chars});sub.subset(font)
        font.flavor='woff';path=assets/f'NotoSansKR-{label}.woff';font.save(path);encoded[label]=base64.b64encode(path.read_bytes()).decode()
    css=''.join('@font-face{font-family:CapstoneKR;src:url(data:font/woff;base64,'+encoded[label]+') format("woff");font-weight:'+str(weight)+';font-style:normal;font-display:swap}' for weight,label in ((400,'Regular'),(700,'Bold')))
    css+='body{font-family:CapstoneKR,"Noto Sans KR","Malgun Gothic",sans-serif}'
    license=(assets/'OFL-NotoSansKR.txt').read_text()
    audits=[]
    for path,text in zip(paths,texts):
        root=html.fromstring(text);visible=' '.join(root.xpath('//text()[not(ancestor::script) and not(ancestor::style)]'))
        bad=re.findall(r'[^.。\n]{0,60}(?:입니다|합니다|습니다|하였다|되었다|했다|한다|된다|이다|있다|없다)[.!?]',visible)
        if bad:raise ValueError('항목형 문구 미적용: '+repr(bad))
        result=text.replace('</style>',css+'</style>',1).replace('</html>','<script id="font-license" type="application/json">'+json.dumps({'license':license},ensure_ascii=False).replace('</','<\\/')+'</script></html>')
        path.write_text(result,encoding='utf-8');audits.append({'file':path.name,'conversational_sentences':bad,'font_embedding':True,'no_visible_font_notice':True})
    write_json(ROOT/'results/usable/html_text_audit.json',audits)


if __name__=='__main__':main()
