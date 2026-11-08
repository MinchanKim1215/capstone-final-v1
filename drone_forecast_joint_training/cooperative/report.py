"""검증·시험 결과 파일로만 생성하는 한국어 통합 결과 설명서."""

from pathlib import Path
import base64, html, json, io
import numpy as np
from swarm_sim.regions import ROOT

LABELS = {
    "nominal": "기존 비행 유지",
    "reactive": "현재 상태 반응",
    "mpc": "예측 기반 MPC",
    "mpc_no_risk": "MPC · 학습 위험 제외",
    "ppo3": "PPO · 36회 학습",
    "ppo8": "PPO · 96회 학습",
    "ppo6": "PPO · 72회 학습",
    "ppo": "학습 정책",
    "ppo_no_risk": "학습 정책 · 위험 제외",
}
FONT = ROOT / "docs/assets/NotoSansKR.ttf"


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + str(x) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def num(x):
    return "—" if x is None else f"{x:,.2f}"


def section(title, body, id):
    return f'<section id="{id}"><h2>{title}</h2>{body}</section>'


def bullet(lines):
    return "<ul>" + "".join("<li>" + s + "</li>" for s in lines) + "</ul>"


def image(path):
    return (
        '<img alt="지역별 실제 실험 결과" class="figure" src="data:image/png;base64,'
        + base64.b64encode(Path(path).read_bytes()).decode()
        + '">'
    )


CSS = """p,li,pre,code,footer{overflow-wrap:anywhere}*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#192b40;font-family:ReportKR,"Malgun Gothic",sans-serif;line-height:1.75}main{max-width:1200px;margin:30px auto;padding:0 26px 60px}.hero{padding:42px 44px;background:#142a45;color:white;border-radius:18px}.eyebrow{font-size:12px;letter-spacing:2px;color:#9cc9ea}h1{font-size:35px;line-height:1.4;margin:12px 0 18px}h2{font-size:23px;margin:0 0 22px}h3{font-size:17px}p{margin:12px 0}.sub{color:#bbccde}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:28px}.card{padding:18px;border:1px solid #476079;border-radius:12px}.card b{display:block;font-size:27px}.card span{font-size:12px;color:#bdcfe1}nav{position:sticky;top:0;display:flex;flex-wrap:wrap;gap:8px;padding:15px 0;background:#f2f5f9f5;z-index:2}nav a{background:white;border:1px solid #d9e1ec;color:#304967;text-decoration:none;padding:7px 12px;border-radius:8px;font-size:13px}section{padding:30px 34px;margin:18px 0;background:#fff;border:1px solid #e0e6ee;border-radius:14px;scroll-margin-top:85px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.6}th,td{padding:12px 13px;border-bottom:1px solid #e0e6ee;text-align:left;vertical-align:top}th{background:#edf3fa;color:#274765;white-space:nowrap}tr:last-child td{border:0}.callout{background:#eaf3fe;border-left:4px solid #2d6cc3;padding:16px 20px;border-radius:6px;margin:20px 0}.note{background:#fff5e9;border-left:4px solid #d78d32;padding:16px 20px;border-radius:6px}.muted,.caption{font-size:12px;color:#5c6e84}.tag{background:#eaf3ff;border-radius:5px;padding:3px 7px;color:#1b5ba3}li{margin:6px 0}.figure{width:100%;margin:14px 0}code{background:#eef3f8;border-radius:4px;padding:2px 6px;font-size:12px}pre{background:#142a45;color:#e2edfa;border-radius:10px;padding:20px;white-space:pre-wrap;font-size:12px;line-height:1.7}select,input,button{font-family:inherit;padding:8px 11px;border:1px solid #c9d5e3;border-radius:7px;background:#fff;color:#263c53}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}.eq{padding:14px 20px;background:#f5f8fc;overflow:auto;font-family:Consolas,monospace;font-size:13px}a{color:#1a5da7}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.statusgood{color:#176943}.statusbad{color:#9b4425}footer{font-size:12px;color:#6b7d91;padding:10px 4px}@media(max-width:800px){main{padding:0 10px}.hero,section{padding:23px}.cards{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}h1{font-size:27px}th,td{padding:8px}}"""


def font_css(text):
    # 사용한 글자만 WOFF로 부분집합 생성. 문서의 시각적 일관성 목적.
    from fontTools.ttLib import TTFont
    from fontTools import subset

    font = TTFont(FONT)
    options = subset.Options()
    options.flavor = "woff"
    sub = subset.Subsetter(options=options)
    sub.populate(text=text)
    sub.subset(font)
    font.flavor = "woff"
    buf = io.BytesIO()
    font.save(buf)
    return (
        "@font-face{font-family:ReportKR;src:url(data:font/woff;base64,"
        + base64.b64encode(buf.getvalue()).decode()
        + ') format("woff");font-weight:100 900;font-display:swap}'
    )


def figures(summary, selection):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    folder = ROOT / "docs/assets/cooperative"
    folder.mkdir(parents=True, exist_ok=True)
    static = ROOT / "cache/cooperative_font.ttf"
    if not static.exists():
        font = TTFont(FONT)
        instantiateVariableFont(font, {"wght": 400}, inplace=True)
        font.save(static)
    fm.fontManager.addfont(str(static))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(static)).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    groups = summary["groups"]
    chosen = selection["selected"]
    keys = list(groups)
    regions = sorted(groups[chosen]["by_region"])
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(regions))
    width = 0.7 / len(keys)
    colors = ["#a4b3c4", "#d49b50", "#4c94aa", "#1f60af", "#8668b6"]
    for k, (key, g) in enumerate(groups.items()):
        vals = [g["by_region"][r]["search_connected_mean"] * 100 for r in regions]
        label = LABELS.get(key, key) + (" · 선정" if key == chosen else "")
        ax.barh(
            y - 0.35 + width / 2 + k * width,
            vals,
            width,
            color=colors[k % len(colors)],
            label=label,
        )
    ax.set_yticks(
        y,
        [
            r.replace("bucheon_jungdong_east", "부천 중동")
            .replace("bucheon_simgokbon", "부천 심곡본")
            .replace("seoul_20261023_b2_", "서울 B2-")
            .replace("seoul_20261023_", "서울 ")
            for r in regions
        ],
    )
    ax.invert_yaxis()
    ax.set(
        xlim=(
            min(
                70,
                min(g["macro"]["search_connected_mean"] * 100 for g in groups.values())
                - 10,
            ),
            100.2,
        ),
        xlabel="수색 3대 연결 유지율 (%)",
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    ax.set_title(
        "미관측 8개 지역 · 지역당 동일 8개 시나리오 평균",
        loc="left",
        weight="bold",
        pad=16,
    )
    fig.tight_layout()
    path = folder / "region_connectivity.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    history = json.loads(
        (ROOT / "results/cooperative/training_history.json").read_text(encoding="utf-8")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    it = [x["iteration"] for x in history]
    axes[0].plot(
        it, [x["connection_mean"] * 100 for x in history], marker="o", color="#1f60af"
    )
    axes[0].set(xlabel="학습 반복", ylabel="학습 실행 연결 유지율 (%)")
    axes[1].plot(it, [x["approx_kl"] for x in history], marker="o", color="#b47735")
    axes[1].set(xlabel="학습 반복", ylabel="정책 갱신 크기 · 근사 KL")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "서로 다른 지역·간섭·방향의 학습 실행 · 동일 조건 성능 추세와 구분",
        x=0.04,
        ha="left",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout()
    train = folder / "policy_training.png"
    fig.savefig(train, dpi=170)
    plt.close(fig)
    return path, train


def decision_figures(summary, selection):
    import matplotlib.pyplot as plt

    folder = ROOT / "docs/assets/cooperative"
    groups = summary["groups"]
    base = groups["nominal"]
    chosen = selection["selected"]
    regions = sorted(base["by_region"])
    others = [k for k in groups if k != "nominal"]
    x = np.arange(len(regions))
    width = 0.24
    colors = {"reactive": "#cd963e", "mpc": "#398d9a", chosen: "#245fa9"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, k in enumerate(others):
        delta = [
            100
            * (
                groups[k]["by_region"][r]["search_connected_mean"]
                - base["by_region"][r]["search_connected_mean"]
            )
            for r in regions
        ]
        ax.bar(
            x + (j - 1) * width,
            delta,
            width,
            label=LABELS.get(k, k),
            color=colors.get(k, "#6e849f"),
        )
    ax.axhline(0, color="#566c82", lw=0.8)
    ax.set_xticks(
        x,
        [
            r.replace("bucheon_jungdong_east", "부천 중동")
            .replace("bucheon_simgokbon", "부천 심곡본")
            .replace("seoul_20261023_b2_", "서울 B2-")
            .replace("seoul_20261023_", "서울 ")
            for r in regions
        ],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("기존 비행 대비 연결 유지율 차이 (%p)")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title(
        "작은 차이의 확대 · 동일 지역에서 기본 비행을 기준으로 비교",
        loc="left",
        weight="bold",
        pad=16,
    )
    fig.tight_layout()
    delta_path = folder / "connectivity_difference.png"
    fig.savefig(delta_path, dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    offsets = {
        "nominal": (10, 5),
        "reactive": (10, -18),
        "mpc": (-125, 12),
        chosen: (10, 10),
    }
    for k, g in groups.items():
        a = g["macro"]["relay_travel_m"] / 1000
        b = g["macro"]["search_connected_mean"] * 100
        ax.scatter(
            [a],
            [b],
            s=90 if k == chosen else 65,
            color=colors.get(k, "#8c9fb4"),
            edgecolors="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            LABELS.get(k, k),
            (a, b),
            xytext=offsets.get(k, (10, 10)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.margins(x=0.25, y=0.25)
    ax.set(xlabel="중계 5대의 총 이동량 평균 (km)", ylabel="연결 유지율 (%)")
    ax.grid(alpha=0.15)
    ax.set_title(
        "연결성과 이동 비용의 상충 · 왼쪽 위 방향이 유리",
        loc="left",
        weight="bold",
        pad=16,
    )
    fig.tight_layout()
    cost_path = folder / "connectivity_movement_tradeoff.png"
    fig.savefig(cost_path, dpi=170)
    plt.close(fig)
    return delta_path, cost_path


def build():
    selection = json.loads(
        (ROOT / "results/cooperative/selection.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (ROOT / "results/cooperative/test_summary.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (ROOT / "configs/cooperative_plan.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (ROOT / "configs/cooperative_evaluation.json").read_text(encoding="utf-8")
    )
    chosen = selection["selected"]
    best = summary["groups"][chosen]
    base = summary["groups"]["nominal"]
    m = best["macro"]
    b = base["macro"]
    delta = 100 * (m["search_connected_mean"] - b["search_connected_mean"])
    figs = figures(summary, selection)
    difference_figs = decision_figures(summary, selection)
    body = '<div class="hero"><div class="eyebrow">UAV COMMUNICATION · CLOSED-LOOP CONTROL</div><h1>통신 장애 예측과<br>학습 기반 재배치 통합</h1><p class="sub">실제 수신 관측 → 학습된 위험 추정 → 이동 선택 → 패킷 전달 결과 반영</p><div class="cards">'
    for value, label in [
        (pct(m["search_connected_mean"]), "선정 방식 · 미관측 지역 연결 유지율"),
        (f"{delta:+.2f}%p", "기존 비행 유지 대비 차이"),
        (pct(m["pdr"]), "종단 간 패킷 전달률"),
        (str(summary["episodes"]) + "회", "공통 시험 시나리오 비교 실행"),
    ]:
        body += f'<div class="card"><b>{value}</b><span>{label}</span></div>'
    body += (
        "</div></div><nav>"
        + "".join(
            f'<a href="#{key}">{name}</a>'
            for key, name in [
                ("result", "비교 결과"),
                ("selection", "선정 기준"),
                ("design", "구현 근거"),
                ("learning", "학습 구성"),
                ("test", "검증"),
                ("details", "개별 결과"),
                ("use", "사용 방법"),
                ("terms", "용어"),
                ("limits", "기타 설명"),
            ]
        )
        + "</nav>"
    )
    body += section(
        "1. 최종 비교 결과",
        '<div class="callout"><b>검증 지역에서 사전 선정한 방식: '
        + html.escape(LABELS.get(chosen, chosen))
        + "</b><p>시험 지역의 결과를 이용한 정책 변경·추가 선정 미실시. 각 지표 개별 제시. 임의 종합 점수 미사용.</p></div>"
        + table(
            [
                "방식",
                "연결 유지율 ↑",
                "PDR ↑",
                "250ms 이내 전달률 ↑",
                "평균 AoI ↓",
                "시간 비율 ↓",
                "중계 이동량 ↓",
                "완주",
            ],
            [
                [
                    LABELS.get(k, k)
                    + (' <span class="tag">선정</span>' if k == chosen else ""),
                    pct(v["macro"]["search_connected_mean"]),
                    pct(v["macro"]["pdr"]),
                    pct(v["macro"]["timely_pdr"]),
                    num(v["macro"]["aoi_mean_s"]) + "s",
                    num(v["macro"]["duration_ratio"]) + "배",
                    num(v["macro"]["relay_travel_m"]) + "m",
                    f"{v['complete_count']}/{v['episodes']}",
                ]
                for k, v in summary["groups"].items()
            ],
        )
        + image(figs[0])
        + image(difference_figs[0])
        + image(difference_figs[1])
        + '<p class="caption">각 실행의 비율을 지역별로 평균한 뒤 8개 지역을 같은 가중치로 평균. 구간·패킷 수를 모두 합친 단일 비율과 구분. 고정 8대·수색 3대·중계 5대·V 40m 기준.</p>',
        "result",
    )
    best_test = max(
        summary["groups"],
        key=lambda k: summary["groups"][k]["macro"]["search_connected_mean"],
    )
    if best_test != chosen:
        body += (
            '<div class="note"><b>검증 선정과 시험 최고 수치의 구분</b><p>시험 연결 유지율 최고: '
            + LABELS.get(best_test, best_test)
            + " "
            + pct(summary["groups"][best_test]["macro"]["search_connected_mean"])
            + ". 검증에서 선정한 "
            + LABELS.get(chosen, chosen)
            + ": "
            + pct(m["search_connected_mean"])
            + ". 시험 결과로 재선정하지 않는 원칙 유지. 학습 정책의 모든 비교군 대비 우위 주장 미포함.</p></div>"
        )
    ci = summary["paired_comparisons"].get("nominal", {})
    if ci:
        r = ci["search_connected_mean"]
        body += section(
            "2. 개선 폭의 불확실성",
            table(
                ["비교", "연결 유지율 차이", "지역 부트스트랩 95% 구간", "해석"],
                [
                    [
                        LABELS.get(chosen, chosen) + " − 기존 비행",
                        f"{r['difference']*100:+.2f}%p",
                        f"[{r['ci95_low']*100:+.2f}, {r['ci95_high']*100:+.2f}]%p",
                        (
                            "구간에 0 포함: 모든 지역의 일관된 우위 확정 불가"
                            if r["ci95_low"] <= 0 <= r["ci95_high"]
                            else "현재 8개 지역 표본에서 개선 방향 지지"
                        ),
                    ]
                ],
            )
            + bullet(
                [
                    "동일 지역·방향·간섭 종류·난수 시드끼리 대응 비교. 지역을 군집 단위로 10,000회 재표집.",
                    "지역 8개에 한정한 기술 통계. 실제 도심 전체·실기체 운용의 신뢰 구간으로 확대 해석 금지.",
                    "연결성 개선과 이동량·임무 시간 증가의 동시 확인.",
                ]
            ),
            "uncertainty",
        )
        body += section(
            "2-1. 비교군별 차이",
            table(
                [
                    "선정 방식 − 비교군",
                    "연결 유지율 차이",
                    "지역 대응 95% 구간",
                    "판단",
                ],
                [
                    [
                        LABELS.get(k, k),
                        f"{v['search_connected_mean']['difference']*100:+.2f}%p",
                        f"[{v['search_connected_mean']['ci95_low']*100:+.2f}, {v['search_connected_mean']['ci95_high']*100:+.2f}]%p",
                        (
                            "차이 방향 확정 어려움"
                            if v["search_connected_mean"]["ci95_low"]
                            <= 0
                            <= v["search_connected_mean"]["ci95_high"]
                            else "현재 지역 표본의 차이 방향 지지"
                        ),
                    ]
                    for k, v in summary["paired_comparisons"].items()
                ],
            ),
            "pairwise",
        )
    body += section(
        "3. 선정 기준과 검증 결과",
        bullet(
            [
                "평가 계획 동결 후 검증 7개 지역·28개 시나리오로 선정.",
                "필수 조건: 28회 임무 완주, 실제 안전 위반 0회, 평균 PDR의 명목 대비 감소 0.5%p 이내, 모든 실행의 시간 비율 1.15 이하.",
                "통과 방식 중 지역 평균 단절 비율 최소 → 적시 전달률 최대 → 중계 이동량 최소 순서. 합산 점수 미사용.",
                "0.5%p·1.15배: 프로젝트의 운영 선택. 논문의 안전 기준·성공 표준으로 인용하지 않음.",
            ]
        )
        + table(
            ["방식", "검증 연결 유지율", "최대 시간 비율", "필수 조건", "미통과 사유"],
            [
                [
                    LABELS.get(k, k),
                    pct(v["macro"]["search_connected_mean"]),
                    num(v["max_duration_ratio"]),
                    (
                        '<span class="statusgood">통과</span>'
                        if v["feasible"]
                        else '<span class="statusbad">미통과</span>'
                    ),
                    " / ".join(v["rejection_reasons"]) or "—",
                ]
                for k, v in selection["candidates"].items()
            ],
        ),
        "selection",
    )
    body += section(
        "4. 전체 동작과 선택 근거",
        table(
            ["구성", "역할", "선택 근거·대안과의 차이"],
            [
                [
                    "최신 시뮬레이터",
                    "정적 지도·비행·RF·패킷·비콘을 3초 제어 루프로 연결",
                    "기존 가중치의 입력 계약과 0.5초 채널을 유지. 팀원 코드의 3초 채널·미래 정보 학습과 분리.",
                ],
                [
                    "LightGBM 예측기",
                    "20초 관측에서 향후 5초 장애·3~7초 새 장애 위험 출력",
                    "기존 45개 지도 비교의 선정 가중치 재사용. 이번 통합에서 GRU·Transformer를 새로 비교·재선정한 결과는 아님.",
                ],
                [
                    "로컬 MPC 비교 방식",
                    "3초·6초의 수신 위치 외삽과 지도 손실로 이동 후보 비교",
                    "학습 전에도 실행 가능한 명시적 기준. 실제 미래 위치·간섭원의 위치/시간 미사용.",
                ],
                [
                    "공유 정책 PPO",
                    "중계별 25개 보정 후보 중 선택, 실제 임무 보상으로 갱신",
                    "PPO [1], 중앙 가치함수·공유 정책의 MAPPO 구조 [2] 참고. 후보 점수 위에 학습 잔차를 더하는 프로젝트 설계. 원 논문의 UAV 성능 재현으로 표현하지 않음.",
                ],
                [
                    "안전 이동기",
                    "건물·경계·속도·최소 분리 거리 검사",
                    "학습 보상에만 의존하지 않고 기존 안전 그래프와 연속 선분 분리 검사 유지. 시뮬레이터의 이상적 충돌 조정 가정.",
                ],
                [
                    "기본 비행·현재 반응·위험 제외 비교",
                    "추가 모듈의 기여와 불필요한 이동 확인",
                    "예측·강화학습을 연결했다는 사실만으로 우위 주장 금지. 검증에서 선택한 뒤 시험으로 확인.",
                ],
            ],
        )
        + '<div class="eq">관측(0.5s) → 예측(3s) → 후보 평가·정책 선택(3s) → 안전 이동 → 채널(0.5s) → TDMA 패킷(5ms) → 다음 관측</div>',
        "design",
    )
    body += section(
        "5. 학습 알고리즘과 입력 경계",
        table(
            ["항목", "현재 구현"],
            [
                [
                    "재배치 주체",
                    "중계 5대. 정책 가중치 공유, 역할별 ID 입력. 수색 3대의 명목 경로 유지.",
                ],
                [
                    "정책 입력",
                    "자기 상태·자기 현재 목표, 성공 수신한 이웃 위치/속도·나이·신호 여유, 지도 기반 후보 특징, 지원되는 학습 위험 점수.",
                ],
                [
                    "행동",
                    "명목 목표에 축별 −40/−20/0/20/40m 보정. 25개 조합. 실제 3초 이동은 안전 그래프의 한 간선으로 제한.",
                ],
                [
                    "가치함수",
                    "현재 전역 위치·채널·큐·임무 진행도. 학습 중 보상/advantage 계산에만 사용. 실제 미래 정보 미입력.",
                ],
                [
                    "학습 방식",
                    "공통 보상, 중앙 가치함수, 로컬 실행 정책. PPO 비율 clip 0.15, GAE λ=0.95, γ=0.995.",
                ],
                [
                    "학습 규모",
                    "30개 학습 지역, 8회 갱신 × 12개 전체 임무 = 96회. 후반 반복에서 간섭 종류·비행 방향 교대.",
                ],
                [
                    "기반 점수 + 잔차",
                    "로컬 지도 기반 후보 점수에 신경망 잔차 추가. 초기 기준 동작 보존 목적. 연속 제어 신호를 합치는 원 논문 [3]의 정확한 재현과 구분.",
                ],
                [
                    "체크포인트",
                    "정책·가치함수·Adam 상태·반복 번호·난수 상태 저장. --resume 재개.",
                ],
                [
                    "실행 단계",
                    "가중치 고정. 추론·재배치만 수행. 비행 중 역전파·서버 연합학습 미실행.",
                ],
                [
                    "E2E 범위",
                    "학습된 예측기와 실제 상호작용으로 학습한 정책의 폐루프 통합. 예측기 고정으로, 예측·제어의 공동 역전파 학습에 해당하지 않음.",
                ],
            ],
        )
        + '<p>훈련 보상: 연결 단절 비용 + 전달 실패/지연 비용 + 이동·수색 지체·보호 개입 비용. 패킷의 생성 구간에 결과를 귀속하여 지연 정산. 최종 성능 선정은 이 보상 합계와 별개.</p><div class="eq">r = 2(C−1) + 0.5(Q−1) − 0.002 − 0.005M − 0.01S − 0.01G<br>Q = 0.5(P + L), L = 성공 패킷의 min(1, 0.25/지연) 합 / 생성 패킷 수</div>'
        + image(figs[1])
        + '<p class="caption">학습 그래프의 매 반복에 다른 시나리오 포함. 곡선 상승만으로 성능 개선 판단 불가. 고정된 28개 검증·64개 시험 시나리오로 정책 성능 확인.</p>',
        "learning",
    )
    body += section(
        "6. 통신 모델과 실험 조건",
        table(
            ["항목", "설정·의미"],
            [
                [
                    "지도·비행",
                    "45개 실제 도시 건물 지도. 400m×400m, 고도 50m 고정, 10m 비행 격자, 최소 분리 5m.",
                ],
                [
                    "경로 손실",
                    "ITU-R P.525 자유공간 손실 + P.526-16 §4.5.1의 평지·지붕 프로파일 Bullington 회절 근사. 원 건물 폴리곤·높이 사용.",
                ],
                [
                    "변동 채널",
                    "시간 상관 AR(1) 섀도잉 σ=1.5dB, τ=4s. 신호/간섭 계산 0.5초 간격.",
                ],
                ["RF", "2.4GHz, 대역폭 1MHz, 송신 10dBm, 잡음 −107dBm."],
                [
                    "외부 간섭",
                    "높이 2m의 외부 간섭원 1개. 지도 내 위치와 −3~3dBm 출력 무작위. 연속/간헐 두 조건. 미래 일정은 정책에 미제공.",
                ],
                [
                    "간헐 간섭",
                    "기존 전체 임무 길이의 15~45% 구간에서 시작, 65~90%에서 종료. 같은 시나리오의 모든 정책에 동일한 절대 시간 적용.",
                ],
                [
                    "패킷 성공",
                    "독립 비트 오류를 가정한 BPSK. 데이터 576B와 역방향 ACK 14B 성공 확률의 곱. 실측 Wi-Fi 곡선 아님.",
                ],
                [
                    "TDMA·비콘",
                    "5ms 이상적 동기 슬롯. 드론당 128B 비콘 1Hz. 매초 8개 슬롯으로 용량 4% 예약.",
                ],
                [
                    "트래픽·라우팅",
                    "수색 드론 사이 총 30패킷/s, TTL 2s, 홉당 최대 4시도, 큐 64개. 0.9 기준 가용 그래프에서 Σ(1/p) 최소 경로. 전역 이상적 라우팅 정보 비용 미반영.",
                ],
                [
                    "예측 관측",
                    "자기 기록과 실제 성공 수신한 데이터 헤더/비콘만 사용. 0.5dB 측정 잡음. 결측은 지원 플래그로 분리.",
                ],
                [
                    "3GPP·ns-3",
                    "TR 36.777 계수와 ns-3 실행 미사용. LoS/NLoS 기하 판정을 3GPP 계수 적용으로 표현하지 않음.",
                ],
                [
                    "분할",
                    "학습 30지역 / 선정 검증 7지역 / 최종 시험 8지역. 인접 중복 공간 그룹 분리 유지.",
                ],
                [
                    "공통 시험",
                    "8지역 × 2방향 × 2간섭 종류 × 2독립 시드 = 방식당 64회 전체 임무. 각 방식에 같은 지도·시드·채널식·비콘 용량.",
                ],
            ],
        ),
        "test",
    )
    body += section(
        "7. 동등성·인과성·안전 검수",
        bullet(
            [
                "기본 비행 90경로: 새 스트리밍 엔진의 보정 0 궤적·명목 목표·단계와 기존 전체 경로의 완전 일치.",
                "테헤란 전체 2,325초·4,650 RF 구간: 관측 마스크·링크 정답·궤적·PDR 일치. 원시 관측 최대 차이 5.69×10⁻¹³.",
                "3초로 정렬된 시점의 예측 확률: 기존 1초 호출 인터페이스와 일치. 제어에 사용하지 않는 수색 드론의 예측 계산 생략. 동일한 비콘 슬롯 비용 유지.",
                "패킷 스트리밍과 기존 일괄 엔진: 시도·ACK·큐·전달·폐기·AoI 일치, 생성=전달+폐기 보존식 검사.",
                "정책 입력 생성: 숨겨진 실제 위치·채널·미래 섀도잉/간섭을 바꾸어도 이미 수신한 입력이 같으면 출력 특징 불변.",
                "이동 안전: 비행 그래프 외 간선·속도 초과·연속 이동 중 5m 미만 분리 시 즉시 실행 실패. 성공 실행만 표에 포함하는 방식 금지; 전체 계획의 완주 수 별도 확인.",
                "학습 기록: 수집 시 행동 확률과 해당 체크포인트의 정책 확률 일치 검사. 모델·지도 지문으로 결과 파일 연결.",
            ]
        ),
        "verification",
    )
    audit_file = ROOT / "results/cooperative/trajectory_integrity_audit.json"
    tests_file = ROOT / "results/cooperative/regression_tests.json"
    if audit_file.exists() and tests_file.exists():
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        checks = json.loads(tests_file.read_text(encoding="utf-8"))
        body += section(
            "7-1. 최종 독립 재계산 결과",
            table(
                ["검사", "결과"],
                [
                    [
                        "회귀·지도 등록 검사",
                        f"서로 다른 {checks['passed']}개 검사 통과",
                    ],
                    [
                        "궤적·패킷·연결성 재계산",
                        f"본 시험·입력 제거 진단 총 {audit['episodes']}회 통과",
                    ],
                    [
                        "연속 이동 최소 분리 거리",
                        f"{audit['minimum_swept_separation_m']:.2f}m · 제한 5m",
                    ],
                    [
                        "최대 실제 이동 속도",
                        f"{audit['maximum_speed_mps']:.2f}m/s · 제한 5m/s",
                    ],
                    ["비행 금지 폴리곤 침범 선분", f"{audit['blocked_segments']}개"],
                    [
                        "실제 수색 셀·연결 성분·단절 시간",
                        "저장된 궤적과 링크로 독립 재계산 후 원본 지표 일치",
                    ],
                ],
            )
            + '<p class="caption">Linux CPU/Python 3.12에서 검사. Windows 실행 파일 제공과 실제 Windows PC 실행 검증의 구분. 검사 원본: results/cooperative/trajectory_integrity_audit.json, regression_tests.json.</p>',
            "finalaudit",
        )
    body += section(
        "7-2. 세부 성능 지표",
        table(
            ["지표"] + [LABELS.get(k, k) for k in summary["groups"]],
            [
                [name]
                + [
                    (
                        pct(v["macro"].get(field))
                        if unit == "%"
                        else num(v["macro"].get(field)) + unit
                    )
                    for v in summary["groups"].values()
                ]
                for name, field, unit in [
                    ("전체 8대 연결률 ↑", "all_drones_connected_mean", "%"),
                    ("수색 쌍 도달률 ↑", "pair_reach_mean", "%"),
                    ("평균 단절 총시간 ↓", "outage_total_s", "s"),
                    ("평균 최장 단절 ↓", "outage_max_s", "s"),
                    ("평균 단절 횟수 ↓", "outage_events", "회"),
                    ("성공 패킷 평균 지연 ↓", "delay_mean_s", "s"),
                    ("성공 패킷 지연 95백분위의 평균 ↓", "delay_p95_s", "s"),
                    ("평균 연결 성분 수 ↓", "mean_components", "개"),
                    ("최대 연결 성분의 기체 비율 ↑", "largest_component_mean", "%"),
                    ("평균 차수 · 연결 이웃 수", "mean_degree", "개"),
                    ("고립 기체 비율 ↓", "isolated_drone_fraction", "%"),
                    ("네트워크 효율 ↑", "network_efficiency_mean", ""),
                    ("연결된 수색 쌍의 최소 홉 수", "search_shortest_hops_mean", "홉"),
                    ("성공 패킷의 실제 홉 수", "actual_delivered_hops_mean", "홉"),
                    (
                        "성공 패킷 중 중계 경유 비율",
                        "delivered_via_relay_fraction",
                        "%",
                    ),
                    ("수색 기체의 대형 거리 RMSE ↓", "search_formation_rmse_m", "m"),
                    ("수색 드론 총 이동량 ↓", "search_travel_m", "m"),
                    ("전체 임무 시간 ↓", "duration_s", "s"),
                    ("관측 가능 셀 수색률 ↑", "observable_coverage", "%"),
                    (
                        "직접 관측 합집합 중 전체 공유 비율 ↑",
                        "shared_observed_fraction",
                        "%",
                    ),
                    (
                        "전체 실외 셀 중 전체 공유 비율 ↑",
                        "information_shared_fraction",
                        "%",
                    ),
                    ("재배치 보정 적용 구간 비율", "relocation_fraction", "%"),
                    ("명목 복귀 보호 개입 구간 비율", "protection_fraction", "%"),
                    ("예측 추론 95백분위의 평균", "forecast_p95_ms", "ms"),
                    ("후보 생성·정책 계산 95백분위의 평균", "decision_p95_ms", "ms"),
                ]
            ],
        )
        + '<p class="caption">지표의 지역 평균. 최소 홉 수: 연결된 쌍에 한정. 실제 홉 수·지연: 도착 패킷에 한정. 중계 경유 비율·차수·홉 수는 높고 낮음만으로 우수성 판단 불가. 대형 RMSE: 수색 3대의 세 쌍 거리와 명목 V 대형 거리의 차이에 대한 제곱평균제곱근. 위치 이동에 따른 장애물 회피 변형 포함.</p>',
        "allmetrics",
    )
    legacy = json.loads(
        (ROOT / "results/advance/report_summary.json").read_text(encoding="utf-8")
    )["original48"]
    body += section(
        "7-3. 예측 모델의 유지 근거",
        table(
            ["통합 전 동일 48회 분류 평가", "AP · 0~1", "재현율", "FPR"],
            [
                [
                    name,
                    f"{legacy[key]['average_precision']:.4f}",
                    pct(legacy[key]["recall"]),
                    pct(legacy[key]["false_positive_rate"]),
                ]
                for name, key in [
                    ("기존 입력 Transformer", "baseline"),
                    ("개선 Transformer", "reference"),
                    ("선정 LightGBM", "new"),
                ]
            ],
        )
        + bullet(
            [
                "기존 선정 가중치 유지. 위 표는 통합 전 예측 비교의 보존 결과이며 이번 재배치 실험에서 새로 측정한 AP와 구분.",
                "LightGBM: 시계열 통계·전파·명령 특징을 사용하는 지도학습 분류기 [5]. 모델 계열의 보편적 우위 대신 현재 데이터의 비교 결과로 채택.",
                "5초 예측 정답: 향후 5초 안에 최소 1초 연속 링크 불가용 발생 여부. 수색군 연결률의 매 0.5초 단절 판정과 구분.",
                "재배치 성공으로 예방된 장애를 곧바로 예측 오경보로 계산하지 않음. 제어 효과는 동일 조건에서 위험 입력을 제외한 비교로 별도 진단.",
                "3초 제어 호출에 기존 1초 경고의 사건 포착률·오경보율을 그대로 적용하지 않음. 이번 제어에는 연속 위험과 지원 플래그 사용.",
            ]
        ),
        "forecast",
    )
    extra_folder = ROOT / "results/cooperative/custom/gangnam_comparison"
    if all((extra_folder / (k + ".json")).exists() for k in ("baseline", "selected")):
        additional = [
            (
                label,
                json.loads((extra_folder / (k + ".json")).read_text(encoding="utf-8")),
            )
            for label, k in [("기존 비행", "baseline"), ("선정 PPO", "selected")]
        ]
        body += section(
            "7-4. 새 조건 실행 기능 검사",
            "<p>강남 · 시드 2026102699 · 동서 수색 · 간헐 간섭. 새 compare 명령의 전체 실행·저장·재생 확인용 추가 사례. 본 시험 256회 평균에 미포함.</p>"
            + table(
                ["방식", "연결 유지율", "PDR", "임무 시간", "완주"],
                [
                    [
                        label,
                        pct(v["search_connected_mean"]),
                        pct(v["pdr"]),
                        num(v["duration_s"]) + "s",
                        "완주" if v["mission_completed"] else "미완주",
                    ]
                    for label, v in additional
                ],
            )
            + '<div class="note">이 추가 사례에서는 선정 PPO의 연결·전달 성능 감소. 평균 개선을 모든 개별 시나리오의 개선으로 해석하지 않음. 결과를 이용한 재선정·가중치 조정 미실시.</div>'
            + '<p class="caption">원본·전체 임무 재생: results/cooperative/custom/gangnam_comparison. 훈련 지역의 새 난수 사례이며 미학습 지역 본 시험과 구분.</p>',
            "customrun",
        )
    raw = []
    for label, rr in summary["rows"].items():
        for r in rr:
            raw.append(dict(group=label, **r))
    body += section(
        "8. 개별 실행 결과",
        '<div class="toolbar"><select id="group"><option value="all">모든 방식</option>'
        + "".join(
            f'<option value="{html.escape(k)}">{html.escape(LABELS.get(k,k))}</option>'
            for k in summary["groups"]
        )
        + '</select><select id="region"><option value="all">모든 지역</option>'
        + "".join(
            f"<option>{html.escape(r)}</option>" for r in sorted(best["by_region"])
        )
        + '</select><select id="kind"><option value="all">모든 간섭</option><option value="continuous">연속 간섭</option><option value="intermittent">간헐 간섭</option></select></div><div class="scroll"><table><thead><tr><th>방식</th><th>지역·방향·시드</th><th>연결 유지율</th><th>PDR</th><th>누적 단절</th><th>최장 단절</th><th>시간 비율</th><th>중계 이동</th><th>완주</th></tr></thead><tbody id="runs"></tbody></table></div><p id="count" class="caption"></p><p>전체 원시 지표: <code>results/cooperative/comparison_all_metrics.csv</code></p>',
        "details",
    )
    ab = ROOT / "results/cooperative/ablation_summary.json"
    if ab.exists():
        ablation = json.loads(ab.read_text(encoding="utf-8"))
        body += section(
            "9. 학습 위험 점수의 추가 효과",
            table(
                ["동일 시험 부분집합", "위험 사용", "위험 제외", "연결 유지율 차이"],
                [
                    [
                        v["label"],
                        pct(v["with_risk"]["macro"]["search_connected_mean"]),
                        pct(v["without_risk"]["macro"]["search_connected_mean"]),
                        f"{100*v['difference']['difference']:+.2f}%p",
                    ]
                    for v in ablation["comparisons"]
                ],
            )
            + bullet(
                [
                    "선정 완료 후 사전 고정한 부분집합의 진단 비교. 이 결과를 이용한 재선정·재학습 미실시.",
                    "MPC 위험 제외: 동일 미래 기하 외삽과 이동 후보 유지, 학습 위험 입력만 제거.",
                    "학습 정책 위험 제외: 같은 정책 가중치의 입력 제거 진단. 위험 없이 별도로 재학습한 PPO와 구분.",
                ]
            ),
            "ablation",
        )
    full_ab = ROOT / "results/cooperative/ablation_full_summary.json"
    if full_ab.exists():
        full = json.loads(full_ab.read_text(encoding="utf-8"))
        body += section(
            "9-1. 전체 64개 시험 조건의 추가 확인",
            table(
                [
                    "고정된 방식",
                    "위험 사용 연결률",
                    "위험 제외 연결률",
                    "차이 · 사용−제외",
                    "지역 대응 95% 구간",
                ],
                [
                    [
                        v["label"],
                        pct(v["with_risk"]["macro"]["search_connected_mean"]),
                        pct(v["without_risk"]["macro"]["search_connected_mean"]),
                        f"{100*v['difference']['difference']:+.2f}%p",
                        f"[{100*v['difference']['ci95_low']:+.2f}, {100*v['difference']['ci95_high']:+.2f}]%p",
                    ]
                    for v in full["comparisons"]
                ],
            )
            + bullet(
                [
                    "미리 고정한 전체 시험 시나리오에 대한 추가 진단. 새 부분집합의 결과 기반 선택 미실시. 선정·가중치·조정 계수 변경 미실시.",
                    "위험 제외 실행 128회: 위 사전 지정 16개 시나리오의 32회를 포함. 중복 합산 금지.",
                    "PPO의 입력 삭제는 학습 분포를 바꾸는 진단. 예측 없이 별도로 학습한 정책의 최적 성능과 동일시하지 않음.",
                    "원본 지표: results/cooperative/risk_ablation_all_metrics.csv. 구간에 0을 포함하면 추가 효과의 방향 확정 유보.",
                ]
            )
            + '<div class="note"><b>예측의 기여와 정책 전체의 개선 구분</b><p>'
            + " ".join(
                (
                    v["label"]
                    + f": 위험 입력의 연결 유지율 차이 {100*v['difference']['difference']:+.2f}%p. "
                    + (
                        "95% 구간에 0 포함. 현재 결과로 예측 입력의 일관된 추가 효과 확정 불가."
                        if v["difference"]["ci95_low"]
                        <= 0
                        <= v["difference"]["ci95_high"]
                        else "현재 지역 표본에서 추가 효과의 방향 지지."
                    )
                )
                for v in full["comparisons"]
            )
            + "</p><p>다음 연구: 위험 입력 없이 별도로 학습한 PPO와 비교, 독립 학습 시드 반복, 학습·검증 지역만으로 예측/행동 결합 개선. 기존 시험 결과를 이용한 현재 가중치 수정 미실시.</p></div>",
            "ablationfull",
        )
    body += section(
        "10. 실행 방법",
        table(
            ["파일·명령", "기능"],
            [
                [
                    "00_환경설정.bat",
                    "Python 3.12 가상환경·CPU PyTorch·의존성 설치. 운영체제 변경 불필요.",
                ],
                [
                    "01_통합_결과.bat",
                    "현재 결과 설명서와 전체 임무 재생 열기. 별도 학습 불필요.",
                ],
                [
                    "02_통합_시뮬레이션.bat",
                    "테헤란 새 시드에서 기존 비행/선정 방식 전체 임무 실행. 새 비교 HTML 자동 열기.",
                ],
                ["03_통합_검사.bat", "지도·가중치·패킷 엔진·통합 회귀 검사."],
                [
                    "cooperate_run.py run --region <키>",
                    "제공된 45개 지역 중 선택. 키 목록: configs/usable_protocol.json.",
                ],
                [
                    "cooperate_run.py compare --region <키> --out <폴더>",
                    "새 지역·시드의 두 방식 비교 후 원본 기록과 replay.html 저장. 공식 시험 결과 변경 없음.",
                ],
                [
                    "cooperate_run.py train --iterations 10 --resume",
                    "저장된 반복부터 학습 계속. 재학습 정책의 성능은 기존 최종 결과와 구분.",
                ],
                [
                    "cooperate_run.py evaluate --split test ...",
                    "동결된 공통 시나리오 재실행 또는 기존 결과 재사용.",
                ],
                [
                    "docs/archive",
                    "통합 전 예측 성능·재생 기록. 현재 재배치 비교 결과와 구분.",
                ],
            ],
        )
        + "<pre>python cooperate_run.py doctor\npython cooperate_run.py compare --region gangnam --seed 2026 --out results/cooperative/custom/gangnam_comparison\npython cooperate_run.py run --region teheran --seed 101 --capture\npython cooperate_run.py verify</pre>",
        "use",
    )
    body += section(
        "11. 핵심 용어와 해석",
        table(
            ["용어", "계산·의미"],
            [
                [
                    "연결 유지율 C",
                    "0.5초마다 수색 3대가 중계를 포함한 같은 연결 성분인지 계산한 0/1 값의 시간 평균. 모든 8대 연결률과 구분.",
                ],
                [
                    "가용 링크",
                    "데이터+역방향 ACK의 성공 확률 p가 양방향 모두 0.9 이상인 링크. 0.9는 채택한 운영 기준. 실제 무선 장비 인증 기준 아님.",
                ],
                [
                    "단절 시간",
                    "수색 3대 전체 연결 조건이 실패한 구간의 길이 합. 단절이 모든 드론의 완전 무통신을 의미하지 않음.",
                ],
                [
                    "PDR",
                    "목적지에 최종 도착한 응용 패킷 수 / 생성 패킷 수. 임무 종료 뒤 최대 2초 정산 포함. 대기 후 복구나 일부 쌍 연결 때문에 연결 유지율과 다를 수 있음.",
                ],
                [
                    "적시 전달률",
                    "생성 후 250ms 이내에 최종 도착한 패킷 수 / 전체 생성 수.",
                ],
                [
                    "AoI",
                    "수신자가 보유한 최신 정보의 생성 시각과 현재 시각의 차이. 수색 드론 사이 방향 쌍 평균. 낮을수록 정보가 신선.",
                ],
                [
                    "시간 비율",
                    "해당 재배치의 전체 임무 시간 / 같은 지도·방향 기본 비행 시간.",
                ],
                [
                    "중계 이동량",
                    "중계 5대의 실제 이동 거리 합. 에너지 소비량의 직접 측정과 구분.",
                ],
                [
                    "MPC",
                    "현재까지 받은 정보에서 가까운 미래의 후보 결과를 계산하고, 다음 3초 행동만 적용한 뒤 다시 계산하는 방식.",
                ],
                [
                    "PPO·중앙 가치함수",
                    "시뮬레이션 보상으로 행동 확률을 학습. 전역 상태는 학습용 가치함수에서만 활용.",
                ],
                [
                    "지도 고정과 일반화",
                    "대형·기체 수를 고정한 상태에서 미학습 지역에 정책을 적용. 임의 대수·기체 성능·고도로의 범용성 증명은 아님.",
                ],
                [
                    "AP·사전 탐지",
                    "순위 예측 성능과 충분한 선행 시간 확보는 별개. 이번 최종 핵심 지표는 실제 재배치 뒤 연결·전달·임무 결과.",
                ],
                [
                    "재현율·FPR",
                    "재현율=TP/(TP+FN): 실제 장애 중 예측으로 잡은 비율. FPR=FP/(FP+TN): 실제 정상 구간 중 오경보 비율. 모든 경고 중 오경보 비율과 구분.",
                ],
                [
                    "대형 RMSE",
                    "수색 3대의 각 쌍 거리와 명목 V 대형 쌍 거리의 차이를 제곱·평균한 뒤 제곱근. 전체 평행이동에 영향을 받지 않는 대형 변형 지표.",
                ],
            ],
        ),
        "terms",
    )
    body += section(
        "12. 기타 설명",
        bullet(
            [
                "시뮬레이션용 통합 구현과 실제 드론 운용 준비도의 구분. 실기체·RF 실측·비행 제어기·분산 MAC 검증 미실시.",
                "이상적 전역 라우팅, 동기 TDMA, 공동 안전 이동기, 정확한 자기 위치, 정적 건물 지도 가정 포함. 순수 분산 실기체 시스템 완성으로 표현하지 않음.",
                "건물 높이의 일부 지역 평균 대체값 포함. 재질·반사·다중경로·도플러·실내 수색·기체 자세·풍속·배터리 정밀 모델 제외.",
                "지도 실외 셀 중 비행 가능한 수색 위치에서 센서로 볼 수 있는 셀을 수색 완료 분모로 사용. 전체 지도 셀의 정보 공유율은 별도 지표.",
                "예측 모델의 관측 부족·현재 열화·명령 부족은 위험 0으로 해석하지 않음. 정책에는 결측 지원 플래그와 실제 수신 상태 제공.",
                "50m 고도·N8·수색3/중계5·V40m 설정의 검증. 이번 결과만으로 다른 드론 대수·대형·중계 비율의 최적성 재주장 금지.",
                "96회 정책 학습은 단일 학습 난수 계열. 시험 시뮬레이션 시드 반복과 독립 정책 재학습 시드 반복은 별개.",
                "기체의 명목 목표·공동 안전 이동·라우팅에 이상적인 정보 가정 포함. 통신을 학습하는 중앙 가치함수와 임무 중 지휘소의 구분. 본 임무의 패킷 목적지는 수색 드론.",
                "수색 완료 뒤 30초 공유 단계와 임무 지체 보호 조건에서 명목 배치 복귀. 실시간 계산 지연을 비행 시간에 추가하는 모델은 미포함.",
                "앞으로의 실기체 적용: 위치·명령 인터페이스 대체, 실제 통신 로그 보정, 비행 제어기 지연과 안전 모니터 연동, 소규모 실내/허가 구역 시험의 별도 검증 필요.",
            ]
        ),
        "limits",
    )
    body += section(
        "13. 방법론 참고문헌",
        bullet(
            [
                '[1] Schulman et al. (2017). <a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a>. §3–5: clipped policy objective, value/entropy loss. PPO 원리의 근거. UAV 조건 최적성의 근거와 구분.',
                '[2] Yu et al. (2022). <a href="https://arxiv.org/html/2103.01955v4">The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games</a>. §3.2–3.3: 중앙 가치함수와 공유 정책. §5: 학습 안정성 관련 실증. 본 구현의 모든 계수·구조를 논문의 권장값으로 간주하지 않음.',
                '[3] Johannink et al. (2019). <a href="https://arxiv.org/abs/1812.03201">Residual Reinforcement Learning for Robot Control</a>. 기존 제어와 학습 잔차 결합의 연구 배경. 본 구현은 이산 후보의 점수 잔차로 별도 설계.',
                '[4] ITU-R <a href="https://www.itu.int/rec/R-REC-P.525/en">P.525</a>, <a href="https://www.itu.int/rec/R-REC-P.526/en">P.526</a>. 기존 경량 채널 모델의 자유공간·회절 근거. 전체 권고안의 정밀 재현과 구분.',
                '[5] Ke et al. (2017). <a href="https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf">LightGBM: A Highly Efficient Gradient Boosting Decision Tree</a>. NeurIPS. 부스팅 트리 학습의 원리 근거. 본 과제의 UAV 성능 우위는 자체 비교 결과로 판단.',
            ]
        ),
        "refs",
    )
    payload = json.dumps({"rows": raw, "labels": LABELS}, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    script = """const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);function render(){const rows=D.rows.filter(r=>($('group').value==='all'||r.group===$('group').value)&&($('region').value==='all'||r.region===$('region').value)&&($('kind').value==='all'||r.kind===$('kind').value));$('runs').innerHTML=rows.map(r=>`<tr><td>${D.labels[r.group]||r.group}</td><td>${r.region}<br>${r.direction} · ${r.kind}<br>seed ${r.seed}</td><td>${(100*r.search_connected_mean).toFixed(2)}%</td><td>${(100*r.pdr).toFixed(2)}%</td><td>${r.outage_total_s.toFixed(1)}s</td><td>${r.outage_max_s.toFixed(1)}s</td><td>${r.duration_ratio.toFixed(3)}</td><td>${r.relay_travel_m.toFixed(0)}m</td><td>${r.mission_completed?'완주':'미완주'}</td></tr>`).join('');$('count').textContent=`표시 ${rows.length}회 / 전체 ${D.rows.length}회`;}for(const k of ['group','region','kind'])$(k).onchange=render;render();"""
    document = (
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>통신 예측·재배치 통합 결과</title><style>'
        + font_css(body)
        + CSS
        + "</style><main>"
        + body
        + '<footer>결과 원본: results/cooperative · 검증 선정과 미관측 지역 시험의 분리</footer></main><script id="data" type="application/json">'
        + payload
        + "</script><script>"
        + script
        + "</script></html>"
    )
    target = ROOT / "docs/통신장애_예측성능_최종검증.html"
    target.write_text(document, encoding="utf8")
    (ROOT / "docs/통합_결과.html").write_text(document, encoding="utf8")
    return target


if __name__ == "__main__":
    print(build())
