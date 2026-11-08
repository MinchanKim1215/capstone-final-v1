"""실제 폐루프 결과의 처음~마지막 구간을 두 방식으로 비교 재생."""

import json, html
from pathlib import Path
import numpy as np
from swarm_sim.regions import ROOT
from cooperative.report import font_css, LABELS


def build(plan=None, selected=None, out=None):
    official = plan is None
    if plan is None:
        plan = json.loads(
            (ROOT / "configs/cooperative_demos.json").read_text(encoding="utf-8")
        )
    if selected is None:
        selected = json.loads(
            (ROOT / "results/cooperative/selection.json").read_text(encoding="utf-8")
        )["selected"]
    cases = []
    for item in plan["cases"]:
        scene = json.loads((ROOT / item["map"]).read_text(encoding="utf-8"))
        runs = []
        for prefix in item["runs"]:
            file = ROOT / prefix
            data = json.loads(
                file.with_suffix(".frames.json").read_text(encoding="utf-8")
            )
            summary = data["summary"]
            frames = data["frames"]
            if official:
                assert summary["mission_completed"], "공식 시연의 수색 완주 결과 필요"
            assert (
                frames[0]["time_s"] == 0
                and frames[-1]["time_s"] == summary["duration_s"]
            )
            with np.load(file.with_suffix(".npz")) as z:
                c = z["C"][:, 0]
                assert len(frames) == len(z["nodes"]) == len(c) // 6 + 1
                # 최근 60초 평균. 모든 0.5초 결과를 계산에 사용하고 3초마다 표시.
                cumulative = np.r_[0, np.cumsum(c)]
                ends = np.arange(6, len(c) + 1, 6)
                starts = np.maximum(0, ends - 120)
                curve = np.c_[
                    ends * 0.5,
                    (cumulative[ends] - cumulative[starts]) / (ends - starts),
                ].tolist()
            # 시각화만 좌표·확률 소수점 축약. 실험 결과 파일의 원래 정밀도 보존.
            for f in frames:
                f["positions"] = np.round(f["positions"], 3).tolist()
                f["risk"] = np.round(f["risk"], 5).tolist()
                f["early"] = np.round(f["early"], 5).tolist()
            runs.append(
                dict(frames=frames, summary=summary, jammer=data["jammer"], curve=curve)
            )
        cases.append(
            dict(
                name=item["name"],
                info=item["info"],
                scene=scene,
                runs=runs,
                seed=item["seed"],
                direction=item["direction"],
                kind=item["kind"],
            )
        )
    payload = json.dumps(
        {"selected_label": LABELS.get(selected, selected),
         "validation_selected":plan.get('validation_selected',official or plan.get('format')=='cooperative-refinement-demos-v2'),
         "cases": cases},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    template = (ROOT / "cooperative/replay_template.html").read_text(encoding="utf-8")
    text = template.replace("__DATA__", payload)
    text = text.replace("__FONT__", font_css(template + payload))
    if out is None:
        out = ROOT / "docs/통신위험_전체임무_최종.html"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not official:
        sources=sorted({str(Path(prefix).parent).replace('\\','/') for item in plan['cases'] for prefix in item['runs']})
        text = text.replace(
            "결과 원본: results/cooperative/demos",
            "결과 원본: "
            + html.escape(' · '.join(sources)),
        )
    out.write_text(text, encoding="utf8")
    if official:
        (ROOT / "docs/통합_전체임무_재생.html").write_text(text, encoding="utf8")
    return out


if __name__ == "__main__":
    print(build())
