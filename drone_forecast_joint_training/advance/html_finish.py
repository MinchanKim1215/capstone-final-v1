"""HTML 항목형 문구·날짜 표기 검사 및 오프라인 표시 처리."""

import base64, json, re
from io import BytesIO
from lxml import html
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools import subset
from swarm_sim.regions import ROOT, read_json, write_json


def main():
    files = [
        ROOT / "docs/통신장애_예측성능_최종검증.html",
        ROOT / "docs/통신위험_전체임무_최종.html",
    ]
    texts = [f.read_text(encoding="utf-8") for f in files]
    # 실행 중 생성물이더라도 현 최종 코드와 같은 학습/평가 소속 메타데이터 유지.
    split_by_id = {
        r["id"]: r["split"]
        for r in read_json(ROOT / "datasets/usable45/collection_manifest.json")["records"]
    }
    match = re.search(r'(<script id="data" type="application/json">)([\s\S]*?)(</script>)', texts[1])
    payload = json.loads(match.group(2))
    for case in payload["cases"]:
        case["summary"]["split"] = split_by_id[case["summary"]["id"]]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    texts[1] = texts[1][:match.start(2)] + encoded + texts[1][match.end(2):]
    write_json(ROOT / "results/advance/replay_summary.json", [c["summary"] for c in payload["cases"]])
    # 재생 지역의 학습/평가 소속 표시. 현재 실행 중 생성물에도 동일 문구 적용.
    notice = "테헤란: 학습 지역 시연. 서울03·부천중동: 평가 지역 시연. 학습 지역 재생과 일반화 검증의 구분."
    if notice not in texts[1]:
        texts[1] = texts[1].replace(
            "<strong>기타 설명</strong><ul>",
            "<strong>기타 설명</strong><ul><li>" + notice + "</li>",
            1,
        )
    texts = [re.sub(r'<style id="embedded-kr">[\s\S]*?</style>', "", s) for s in texts]
    texts = [
        re.sub(
            r'<script id="font-license" type="application/json">[\s\S]*?</script>',
            "",
            s,
        )
        for s in texts
    ]
    chars = {ord(c) for c in "".join(texts)}
    assets = ROOT / "docs/assets"
    css = ""
    for weight in [400, 700]:
        f = TTFont(assets / "NotoSansKR.ttf")
        instantiateVariableFont(f, {"wght": weight}, inplace=True)
        options = subset.Options()
        options.flavor = "woff"
        ss = subset.Subsetter(options=options)
        ss.populate(unicodes=chars)
        ss.subset(f)
        f.flavor = "woff"
        out = BytesIO()
        f.save(out)
        css += (
            "@font-face{font-family:CapstoneKR;src:url(data:font/woff;base64,"
            + base64.b64encode(out.getvalue()).decode()
            + ') format("woff");font-weight:'
            + str(weight)
            + ";font-display:swap}"
        )
    css += 'body{font-family:CapstoneKR,"Malgun Gothic",sans-serif}'
    license = (assets / "OFL-NotoSansKR.txt").read_text(encoding="utf-8")
    audit = []
    for file, text in zip(files, texts):
        root = html.fromstring(text)
        visible = " ".join(
            root.xpath("//text()[not(ancestor::script) and not(ancestor::style)]")
        )
        bad = re.findall(
            r"[^.。\n]{0,60}(?:입니다|합니다|습니다|하였다|되었다|했다|한다|된다|이다|있다|없다)[.!?]",
            visible,
        )
        if bad:
            raise ValueError("항목형 문구 미적용: " + repr(bad))
        for word in ["교수님께 설명할 때", "제작일", "내장 한글 글꼴"]:
            assert word not in visible
        text = text.replace(
            "</style>", '</style><style id="embedded-kr">' + css + "</style>", 1
        )
        text = text.replace(
            "</html>",
            '<script id="font-license" type="application/json">'
            + json.dumps({"license": license}, ensure_ascii=False).replace("</", "<\\/")
            + "</script></html>",
        )
        file.write_text(text, encoding="utf-8")
        audit.append(
            {
                "file": file.name,
                "conversational_sentences": bad,
                "no_creation_date": True,
                "no_visible_font_notice": True,
            }
        )
    write_json(ROOT / "results/advance/html_audit.json", audit)


if __name__ == "__main__":
    main()
