"""최종 예측기의 실제 observe 인터페이스를 연결한 세 전체 임무 재생."""

import csv, json, time
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json
from usable.data import collection, load_observation
from advance.features import observed_commands
from advance.predictor import LinkRiskPredictor
from advance.data import RESULT
from advance.train import persist

STATUS = {
    "normal": 1,
    "watch": 2,
    "warning": 3,
    "stale_observations": 4,
    "currently_degraded": 5,
    "insufficient_observations": 6,
    "insufficient_intent_observations": 7,
}


def template():
    text = (ROOT / "usable/replay_template.html").read_text(encoding="utf-8")
    text = text.replace(".b5,.b4,.b6{", ".b5,.b4,.b6,.b7{")
    text = text.replace(
        "경고: 운용 진입 문턱 도달. 낮은 해제 문턱까지 상태 유지.",
        "경고: 검증에서 선정한 출력의 진입 문턱 도달. 낮은 해제 문턱까지 유지. 표시한 5초 위험의 단순 순위와 구분.",
    )
    text = text.replace(
        ".stats{display:flex;gap:10px}",
        ".stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}",
    )
    text = text.replace(
        '<span><i style="background:#333e50"></i>50m 이상 건물</span>',
        '<span><i style="background:#7955b8"></i>8초 사전 주의</span><span><i style="background:#333e50"></i>50m 이상 건물</span>',
    )
    replacements = [
        ("학습된 예측기 연결", "최종 선정 예측기 연결"),
        (
            '<select id="case"></select>',
            '<select id="case"></select><select id="scoreview"><option value="risk">향후 5초 위험</option><option value="early">3–4초 뒤 새 장애</option><option value="early8">3–7초 뒤 새 장애</option></select>',
        ),
        ("<th>5초 위험</th>", '<th id="scoretitle">5초 위험</th>'),
        ("'관측 부족'],colors=", "'관측 부족','명령 관측 부족'],colors="),
        ("'#b6bec9'];", "'#b6bec9','#b6bec9'];"),
        ("p=c.risk[t][ix]", "p=c[$('scoreview').value][t][ix],e=!!c.prewarning[t][ix]"),
        ("rows.push({i,j,s,p})", "rows.push({i,j,s,p,e})"),
        ("r.s===0||r.s===4||r.s===6", "r.s===0||r.s===4||r.s===6||r.s===7"),
        (
            "$('truth').onchange=render;",
            "$('truth').onchange=render;$('scoreview').onchange=render;$('showearly').onchange=render;",
        ),
        (
            '<div class="stat"><b id="unkcount">',
            '<div class="stat"><b id="earlycount">0</b><span>8초 사전 주의 링크</span></div><div class="stat"><b id="unkcount">',
        ),
    ]
    for old, new in replacements:
        if old not in text:
            raise ValueError("재생 양식 기준 문자열 누락: " + old)
        text = text.replace(old, new, 1)
    target = "for(const r of rows.filter(r=>r.s>=1&&r.s<=3)"
    if target not in text:
        raise ValueError("링크 그리기 구간 누락")
    text = text.replace(
        target,
        "if($('showearly').checked)for(const r of rows.filter(r=>r.e)){ctx.strokeStyle='#7955b8cc';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(...xy(c.positions[t][r.i]));ctx.lineTo(...xy(c.positions[t][r.j]));ctx.stroke()}\n"
        + target,
        1,
    )
    target = "$('watchcount').textContent="
    text = text.replace(
        target,
        "$('earlycount').textContent=rows.filter(r=>r.e).length;$('scoretitle').textContent=({risk:'5초 위험',early:'3–4초 새 장애',early8:'3–7초 새 장애'})[$('scoreview').value];"
        + target,
        1,
    )
    target = '<input type="checkbox" id="truth">'
    text = text.replace(
        target,
        '<input type="checkbox" id="showearly"> 8초 사전 주의 표시</label><br><label style="font-size:12px">'
        + target,
        1,
    )
    text = text.replace(
        "<strong>기타 설명</strong><ul>",
        "<strong>기타 설명</strong><ul><li>테헤란: 학습 지역 시연. 서울03·부천중동: 평가 지역 시연. 학습 지역 재생과 일반화 검증의 구분.</li><li>3–4초·3–7초: 해당 선행 구간에서 시작할 새 단절의 위험. 5초·8초 경고 계약의 분리.</li><li>현재 목표: 자기 발행 목표와 실제 수신 비콘의 이웃 목표. 실제 미래 좌표와 별개.</li><li>마지막 10초: 예측 출력 재생, 주 분류 평가의 미래 정답 확보 범위 외.</li>",
        1,
    )
    return text


def case(r, entry):
    x, m = load_observation(r)
    goal = observed_commands(r, x, m)
    n = x.shape[1]
    folder = ROOT / "model/advance/selected"
    predictor = LinkRiskPredictor(folder / "best.pt", ROOT / entry["path"])
    with np.load(ROOT / r["path"] / "truth.npz") as z:
        xyz = z["positions_start_m"]
        up = z["link_available"]
        jam = z["jammer_active"]
        connected = z["search_connected"]
    null = [None] * (n * n)
    positions = [np.round(xyz[0, :, :2], 1).tolist()]
    risk = [null.copy()]
    early = [null.copy()]
    early8 = [null.copy()]
    status = [[0] * (n * n)]
    prewarning = [[0] * (n * n)]
    truth = [up[0].astype(int).ravel().tolist()]
    jams = [int(jam[0])]
    conn = [int(connected[0])]
    rows = []
    started = time.perf_counter()
    for t in range(len(x)):
        out = predictor.observe((t + 1) * 0.5, x[t], m[t], goal[t])
        if (t + 1) % 2:
            continue
        p = null.copy()
        e = null.copy()
        ee = null.copy()
        s = [0] * (n * n)
        pre = [0] * (n * n)
        for link in out["links"]:
            k = link["source"] * n + link["peer"]
            p[k] = round(link["risk_5s"], 5) if link["risk_5s"] is not None else None
            e[k] = (
                round(link["risk_onset_3to4s"], 5)
                if link["risk_onset_3to4s"] is not None
                else None
            )
            ee[k] = (
                round(link["risk_onset_3to7s"], 5)
                if link["risk_onset_3to7s"] is not None
                else None
            )
            s[k] = STATUS[link["status"]]
            pre[k] = int(link["early_warning_8s"])
            rows.append({"time_s": (t + 1) * 0.5, **link})
        end = x[t, np.arange(n), (np.arange(n) + 1) % n, 12:14]
        positions.append(np.round(end, 1).tolist())
        risk.append(p)
        early.append(e)
        early8.append(ee)
        status.append(s)
        prewarning.append(pre)
        truth.append(up[t].astype(int).ravel().tolist())
        jams.append(int(jam[t]))
        conn.append(int(connected[t]))
    summary = {
        "id": r["id"],
        "region": r["region"],
        "split": r["split"],
        "duration_s": r["metadata"]["duration_s"],
        "rendered_start_s": 0,
        "rendered_end_s": len(positions) - 1,
        "complete_mission": r["metadata"]["mission"]["mission_completed"],
        "stream_observe_calls": len(x),
        "elapsed_s": time.perf_counter() - started,
        "prediction_control_repositioning": False,
    }
    assert summary["duration_s"] == summary["rendered_end_s"]
    dest = RESULT / "replays"
    dest.mkdir(exist_ok=True)
    with (dest / (r["id"] + "_risk.csv")).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scene = read_json(ROOT / entry["path"])
    return {
        "name": entry["name"],
        "direction": r["metadata"]["direction"],
        "kind": r["metadata"]["kind"],
        "summary": summary,
        "positions": positions,
        "risk": risk,
        "early": early,
        "early8": early8,
        "status": status,
        "prewarning": prewarning,
        "truth": truth,
        "jammer": jams,
        "connected": conn,
        "buildings": scene["buildings"],
        "no_fly_zones": scene["no_fly_zones"],
    }


def main():
    manifest, p = collection()
    data = []
    for key, direction, kind in [
        ("teheran", "east_west", "intermittent"),
        ("seoul_20261023_03", "north_south", "intermittent"),
        ("bucheon_jungdong_east", "east_west", "continuous"),
    ]:
        r = next(
            r
            for r in manifest["records"]
            if r["region"] == key
            and r["metadata"]["direction"] == direction
            and r["metadata"]["kind"] == kind
        )
        entry = next(e for e in p["maps"] if e["key"] == key)
        c = case(r, entry)
        data.append(c)
        print("full replay", c["summary"], flush=True)
    payload = {"cases": data, "n_drones": 8, "n_search": 3}
    text = template().replace(
        "__DATA__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        ),
    )
    (ROOT / "docs/통신위험_전체임무_최종.html").write_text(text, encoding="utf-8")
    write_json(RESULT / "replay_summary.json", [r["summary"] for r in data])
    persist("full_mission_replays_complete")


if __name__ == "__main__":
    main()
