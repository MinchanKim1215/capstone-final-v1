"""실제 결과 JSON과 점수 배열만 참조하는 항목형 한국어 설명서."""

import base64, html, json, re
from pathlib import Path
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json
from advance.data import RESULT
from advance.train import RUNS


def n(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def p(v, d=1):
    return "—" if v is None else f"{100*v:.{d}f}%"


def esc(v):
    return html.escape(str(v))


def table(head, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + str(x) + "</th>" for x in head)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(v) + "</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def bullets(rows):
    return "<ul>" + "".join("<li>" + x + "</li>" for x in rows) + "</ul>"


def section(id, title, body):
    return '<section id="' + id + '"><h2>' + title + "</h2>" + body + "</section>"


def figure(path, caption):
    return (
        '<figure><img alt="'
        + esc(caption)
        + '" src="data:image/png;base64,'
        + base64.b64encode(path.read_bytes()).decode()
        + '"><figcaption>'
        + caption
        + "</figcaption></figure>"
    )


def plots(selection, r):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    assets = ROOT / "docs/assets"
    font = ROOT / "cache/advance_report_font.ttf"
    if not font.exists():
        f = TTFont(assets / "NotoSansKR.ttf")
        instantiateVariableFont(f, {"wght": 400}, inplace=True)
        f.save(font)
    fm.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": fm.FontProperties(fname=str(font)).get_name(),
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    out = {}
    with np.load(RESULT / "regression_aggregate.npz") as z:
        y = z["y"]
        a = z["new"]
        b = z["baseline"]
        t = z["reference"]
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(10.8, 5))
    series = [
        (b, "기존 Transformer", "#9aa9bc"),
        (t, "개선 Transformer", "#ba7c47"),
        (a, "최종 선정 예측기", "#195cb1"),
    ]
    for score, label, color in series:
        precision, recall, _ = precision_recall_curve(y, score)
        take = np.unique(
            np.linspace(0, len(precision) - 1, min(2200, len(precision))).astype(int)
        )
        ax.plot(recall[take], precision[take], label=label, color=color, lw=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="재현율", ylabel="정밀도")
    ax.set_title("전체 48개 비행 · 정밀도–재현율", loc="left", weight="bold", pad=14)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    file = assets / "advance_pr.png"
    fig.savefig(file, dpi=175)
    plt.close(fig)
    out["pr"] = file
    rows = selection["all_candidates"]
    base = read_json(RESULT / "baseline_validation_metrics.json")["main"][
        "average_precision"
    ]
    names = ["기존 Transformer"]
    scores = [base]
    physical = RESULT / "physical_validation_probe.json"
    if physical.exists():
        names.append("지도·현재 목표 외삽 기준")
        scores.append(read_json(physical)["5s_AP"])
    for row in rows:
        if row["checkpoint"] != "best.pt":
            continue
        names.append(
            ("LightGBM " + row["run"].split("_")[0][4:] + "리프")
            if row["family"] == "GBDT"
            else "개선 Transformer"
        )
        scores.append(row["ap5"])
    fig, ax = plt.subplots(figsize=(10.8, 4))
    q = np.arange(len(scores))
    ax.barh(
        q,
        scores,
        color=["#9aa9bc"]
        + [
            (
                "#195cb1"
                if "LightGBM" in v
                else "#ba7c47" if "Transformer" in v else "#82a5a0"
            )
            for v in names[1:]
        ],
        height=0.62,
    )
    ax.set_yticks(q, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP · 공통 검증 창 57,344개")
    ax.set_title("모델 선택용 검증 비교", loc="left", weight="bold", pad=14)
    for i, v in enumerate(scores):
        ax.text(v + 0.008, i, n(v), va="center")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    file = assets / "advance_validation.png"
    fig.savefig(file, dpi=175)
    plt.close(fig)
    out["validation"] = file
    maps = {
        e["key"]: e["name"]
        for e in read_json(ROOT / "configs/usable_protocol.json")["maps"]
    }
    keys = sorted(r["new"]["by_region"])
    q = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    ax.barh(
        q + 0.18,
        [r["baseline_retuned"]["by_region"][k]["average_precision"] for k in keys],
        0.32,
        label="기존 Transformer",
        color="#9aa9bc",
    )
    ax.barh(
        q - 0.18,
        [r["new"]["by_region"][k]["average_precision"] for k in keys],
        0.32,
        label="최종 선정",
        color="#195cb1",
    )
    ax.set_yticks(q, [maps[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP")
    ax.set_title("지역별 전체 평가", loc="left", weight="bold", pad=14)
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    file = assets / "advance_regions.png"
    fig.savefig(file, dpi=175)
    plt.close(fig)
    out["regions"] = file
    history = read_json(RUNS / "transformer_s51/history.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    for ax, key, title in [
        (axes[0], "validation_ap", "향후 5초 위험 AP"),
        (axes[1], "early_ap", "3–4초 새 단절 AP"),
    ]:
        ax.plot(
            [v["step"] for v in history],
            [v[key] for v in history],
            color="#ba7c47",
            marker="o",
            ms=3,
        )
        ax.set_xlabel("학습 갱신 수")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    file = assets / "advance_learning.png"
    fig.savefig(file, dpi=175)
    plt.close(fig)
    out["learning"] = file
    return out


def alarm_table(result, horizon):
    rows = []
    for label, key in [
        ("기존 · 동일 기준 재조정", f"baseline_retuned_{horizon}s"),
        ("최종 선정", f"new_{horizon}s"),
    ]:
        a = result["alarms"][key]
        rows.append(
            [
                label,
                p(a["first_alert_event_recall_3s"], 2),
                p(a["first_alert_precision"]),
                n(a["false_alerts_per_drone_hour"], 2),
                f"{a['first_alerts']:,}",
                f"{a['all_new_directed_events']:,}",
            ]
        )
    return table(
        [
            "모형",
            "첫 경고 ≥3초 선행 포착",
            "첫 경고 정밀도",
            "오경보·늦은 알림/드론·시간",
            "첫 경고 수",
            "새 사건 수",
        ],
        rows,
    )


def main():
    selection = read_json(RESULT / "selection.json")
    r = read_json(RESULT / "regression_evaluation.json")
    c = read_json(RESULT / "confirmation_evaluation.json")
    runtime = read_json(RESULT / "runtime_verification.json")
    audit = read_json(RESULT / "final_audit.json")
    fig = plots(selection, r)
    choice = selection["selected"]
    family = "LightGBM" if choice["family"] == "GBDT" else "Transformer"
    old = r["baseline_retuned"]["all"]
    new = r["new"]["all"]
    ref = r["transformer_reference"]["all"]
    oa = r["alarms"]["baseline_retuned_5s"]
    na = r["alarms"]["new_5s"]
    cards = [
        (
            "동일 48비행 AP",
            n(old["average_precision"]) + " → " + n(new["average_precision"]),
            "확률 보정 후 위험 순위",
        ),
        (
            "창 단위 재현율",
            p(old["recall"]) + " → " + p(new["recall"]),
            "같은 검증 선정 기준",
        ),
        (
            "첫 경고 3초 선행 포착",
            p(oa["first_alert_event_recall_3s"], 2)
            + " → "
            + p(na["first_alert_event_recall_3s"], 2),
            "5초 경고 계약 · 사건 단위",
        ),
        (
            "새 난수 24비행 AP",
            n(c["baseline_retuned"]["all"]["average_precision"])
            + " → "
            + n(c["new"]["all"]["average_precision"]),
            "모델·문턱 동결 후 확인",
        ),
    ]
    body = (
        '<div class="metrics">'
        + "".join(
            '<div class="metric"><span>'
            + k
            + "</span><strong>"
            + v
            + "</strong><small>"
            + s
            + "</small></div>"
            for k, v, s in cards
        )
        + "</div>"
    )
    warnings = []
    if na["first_alert_event_recall_3s"] < 0.5:
        warnings.append(
            f'기존 48비행의 5초 정책: 첫 경고 3초 선행 포착 {p(na["first_alert_event_recall_3s"],2)}. 미포착 사건 비중 {p(1-na["first_alert_event_recall_3s"],2)}. 단독 사전 재배치 트리거의 충분성 미확인. 연속 위험 입력과 8초 사전 주의의 후속 제어 검증 필요.'
        )
    for name, result in [("기존 48비행", r), ("새 난수 24비행", c)]:
        point = result["new"]["all"]
        alarm = result["alarms"]["new_5s"]
        if point["precision"] is None or point["precision"] < 0.7:
            warnings.append(
                name
                + ": 검증 창 정밀도 70% 목표의 시험 미충족. 시험 후 문턱 재조정 미실시."
            )
        if (
            alarm["first_alert_precision"] is None
            or alarm["first_alert_precision"] < 0.8
        ):
            warnings.append(
                name + ": 첫 경고 정밀도 80% 목표의 시험 미충족 또는 경고 미선정."
            )
        if alarm["false_alerts_per_drone_hour"] > 5:
            warnings.append(name + ": 오경보 5회/드론·시간 목표의 시험 초과.")
        early = result["alarms"]["new_8s"]
        if (
            early["first_alert_precision"] is None
            or early["first_alert_precision"] < 0.8
        ):
            warnings.append(
                name
                + ": 별도 8초 정책의 첫 경고 정밀도 80% 목표 미충족 또는 경고 미선정."
            )
        if early["false_alerts_per_drone_hour"] > 5:
            warnings.append(name + ": 별도 8초 정책의 오경보 5회/드론·시간 목표 초과.")
    if warnings:
        body += section("findings", "확인 필요 항목", bullets(warnings))
    rows = []
    for label, v in [
        ("기존 Transformer", old),
        ("개선 Transformer", ref),
        ("최종 " + family, new),
    ]:
        rows.append(
            [
                label,
                n(v["average_precision"]),
                p(v["precision"]),
                p(v["recall"]),
                p(v["false_positive_rate"], 2),
                n(v["brier"], 4),
            ]
        )
    text = table(["모형", "AP ↑", "정밀도 ↑", "재현율 ↑", "FPR ↓", "Brier ↓"], rows)
    text += f'<p>공통 평가 창 {r["samples"]:,}개 · 위험 창 비율 {p(r["positive_fraction"],2)}. 같은 비행·과거 관측·미래 5초 정답. 같은 114,688개 검증 보정 창과 단조 로지스틱 보정, 검증 정밀도 70% 이상·FPR 2% 이하의 동작점 규칙. 시험에서의 실제 달성치는 위 표.</p>'
    text += figure(
        fig["pr"],
        "전체 점수로 계산한 PR 곡선. 표시 점만 축소, AP 계산에는 모든 평가 창 사용.",
    )
    text += "<h3>동결 후 새 난수 24비행</h3>" + table(
        ["모형", "AP ↑", "정밀도 ↑", "재현율 ↑", "FPR ↓", "Brier ↓"],
        [
            [
                label,
                n(v["average_precision"]),
                p(v["precision"]),
                p(v["recall"]),
                p(v["false_positive_rate"], 2),
                n(v["brier"], 4),
            ]
            for label, v in [
                ("기존 Transformer", c["baseline_retuned"]["all"]),
                ("개선 Transformer", c["transformer_reference"]["all"]),
                ("최종 " + family, c["new"]["all"]),
            ]
        ],
    )
    text += f'<p class="caption">공통 평가 창 {c["samples"]:,}개 · 위험 창 비율 {p(c["positive_fraction"],2)}. 기존 평가 8지역의 새 채널·간섭·패킷·비콘 난수. 동일하게 고정한 가중치·확률 보정·문턱 적용. 새 지역 일반화 검증과 구분.</p>'
    rt = r["runtime"]
    text += "<h3>실행 지원 관측 조건의 비교</h3>" + table(
        ["모형", "AP", "정밀도", "재현율", "FPR"],
        [
            [
                label,
                n(v["average_precision"]),
                p(v["precision"]),
                p(v["recall"]),
                p(v["false_positive_rate"], 2),
            ]
            for label, v in [
                ("기존 · 동일 보정", rt["baseline"]),
                ("최종 선정", rt["new"]),
            ]
        ],
    )
    text += f'<p class="caption">관측 이력 충족 창 중 신선한 양방향 측정·양수 측정 여유를 충족한 {rt["operational_windows"]:,}개 창({p(rt["operational_fraction"])}). 실행 대상 선별에 실제 현재 가용성 정답 미사용. 관측 미지원은 확률 None 및 상태 플래그로 유지.</p>'
    body += section("performance", "1. 예측 성능 비교", text)
    text = (
        "<h3>기존 48비행 · 5초 경고 계약</h3>"
        + alarm_table(r, 5)
        + "<h3>새 난수 24비행 · 5초 경고 계약</h3>"
        + alarm_table(c, 5)
        + "<h3>별도 8초 사전 주의 · 새 난수 24비행</h3>"
        + alarm_table(c, 8)
    )
    text += bullets(
        [
            "5초 위험: 향후 5초 안의 최소 1초 연속 링크 불가용.",
            "5초 새 사건에서 3초 대응 여유 확보 시 유효 선행 구간 3~4초. 8초 계약의 유효 구간 3~7초.",
            "3초 선행시간: 3초 이동 제어 주기를 참고한 프로젝트 비교값. 실제 회피 완료 시간의 추가 검증 필요.",
            "경고 선택: 검증 첫 경고 정밀도 80% 이상·오경보 5회/드론·시간 이하·첫 경고 30회 이상. 그 안에서 3초 선행 사건 포착 최대화.",
            "같은 문턱 탐색·히스테리시스를 기존 모델에도 적용. 20회·60회 예산은 원본 JSON의 별도 대안 비교, 기본 기준과 구분.",
            "연속 경고의 첫 시각을 한 알림으로 집계. 사건 재현율의 중복 가산 제외.",
            "사건 분모: 시작 20초 이후 모든 1초 이상 새 단절. 관측 미지원 사건 포함. 마지막 10초의 결정 창 제외와 해당 구간 사건의 분모 포함은 기존 보수적 정의 유지.",
            "방향별 관측 드론 기준의 사건 집계. 두 방향과 겹치는 창의 통계적 상관 존재.",
        ]
    )
    body += section("alerts", "2. 조기 경고의 실제 성과", text)
    duration_file = RESULT / "confirmation_duration_analysis.json"
    if duration_file.exists():
        d = read_json(duration_file)
        rows = []
        for minimum in [1, 3, 5]:
            values = {
                v["model"]: v
                for v in d["rows"]
                if v["horizon_s"] == 8 and v["actual_outage_minimum_s"] == minimum
            }
            rows.append(
                [
                    f"{minimum}초 이상",
                    f"{values['new']['events']:,}",
                    p(values["baseline"]["first_alert_3s_recall"], 2),
                    p(values["new"]["first_alert_3s_recall"], 2),
                ]
            )
        extra = table(
            [
                "실제 단절 길이",
                "새 사건 수",
                "기존 첫 경고 ≥3초 포착",
                "선정 첫 경고 ≥3초 포착",
            ],
            rows,
        )
        extra += '<p class="caption">새 난수 24비행, 고정 8초 정책의 실제 지속시간별 포착 분해. 원래 1초 정답·가중치·경고 문턱 유지. 긴 단절 전용 모델 성능 또는 새 합격 기준으로 해석하지 않는 범위. 전체 첫 경고 정밀도·오경보는 앞 표의 1초 계약 기준.</p>'
        body += section("duration", "실제 단절 길이별 추가 확인", extra)
    rows = [
        [
            v["run"],
            v["checkpoint"],
            n(v["ap5"]),
            n(v["early_ap"]),
            "선정" if v == choice else "비교",
        ]
        for v in selection["all_candidates"]
    ]
    text = f'<p><b>최종 방법: {family}</b> · {esc(choice["run"])} · 대표 시드 51. 최고 검증 5초 AP에서 0.005 이내 후보 중 3~4초 새 단절 AP 우선. 지표 합산 점수 미사용. 허용폭·우선순위는 프로젝트 사전 설정.</p>'
    text += table(
        ["후보", "가중치", "5초 검증 AP", "같은 가중치의 3~4초 AP", "용도"], rows
    ) + figure(fig["validation"], "같은 검증 창 57,344개의 비교. 시험 결과를 이용한 가중치 선택 제외.")
    prevalence = read_json(ROOT / "datasets/advance/select/bank.json")["target_means"]
    text += f'<p class="caption">모델 선택 창의 5초 위험 비율 {p(prevalence[0],2)}, 3–4초 새 단절 비율 {p(prevalence[3],2)}. 무정보 순위의 AP 기준선은 해당 위험 비율에 근접. 서로 다른 정답의 AP를 같은 정확도 백분율로 비교하지 않는 해석.</p>'
    text += '<p class="caption">31리프·63리프의 5초 AP는 거의 같은 수준. 사전에 고정한 새 단절 AP 우선 규칙에 따른 63리프 대표 선택. 모든 조건의 통계적 우월성 증명과 구분.</p>'
    text += table(
        ["구성", "역할·선택 이유"],
        [
            [
                "현재 통신 기록",
                "신호 세기·양방향 여유·ACK·결측/나이·현재 자기 상태. 현장에서 이용 가능한 관측",
            ],
            [
                "지도 중간 시점",
                "0.5~8초의 13시점 외삽. 끝점 사이의 건물 차폐 변화 표현",
            ],
            [
                "전파 보조 입력",
                "측정 전력−SINR·정적 지도 잔차·시간 상관 가설. 잠재 실제 채널 조회 제외",
            ],
            [
                "현재 목표 공유",
                "현재 속도만으로 구분하기 어려운 정지·회전·진행 의도 표현. 아직 발행되지 않은 목표 제외",
            ],
            [
                "LightGBM",
                "과거 통계 660개와 보조 137개로 비선형 조건 학습. 4개 정답별 같은 방법 적용. 채택 근거는 공통 검증 결과, 서로 다른 방법의 확률 평균 미사용",
            ],
            [
                "개선 Transformer",
                "기존 인코더와 최신 관측 유지, 지도/목표·통계 잔차 경로 추가. 같은 입력·정답의 신경망 비교 및 후속 공동 학습 참고",
            ],
        ],
    )
    text += '<p class="caption">새 구성은 입력·학습 기록·학습 구조를 함께 변경한 통합 개선. 기존 Transformer와의 차이를 특정 한 요소의 단독 인과 효과로 해석하지 않는 범위. Transformer 원 논문은 번역 실험, LightGBM 원 논문은 일반 데이터 실험으로 본 UAV 조건의 직접 우수성 증거와 구분.</p>'
    gain_file = RESULT / "training_feature_gain.json"
    if gain_file.exists():
        gains = read_json(gain_file)
        if gains["checkpoint_sha256"] != choice["checkpoint_sha256"]:
            raise ValueError("분할 이득 설명과 최종 가중치 불일치")
        g0 = gains["heads"]["0"]["category_gain_fraction"]
        g3 = gains["heads"]["3"]["category_gain_fraction"]
        order = sorted(g0, key=lambda k: g0[k], reverse=True)
        text += "<h3>선정 트리의 입력 사용 내역</h3>" + table(
            ["입력 묶음", "5초 위험의 분할 이득 비중", "3–4초 새 단절의 분할 이득 비중"],
            [[k, p(g0[k]), p(g3.get(k, 0))] for k in order],
        )
        text += '<p class="caption">학습 중 선택한 트리 분할의 손실 감소량 합계. 각 출력 안에서 정규화한 설명용 비중. 평가 성능의 기여도·인과 효과·입력 제거 실험과 구분. 상관된 입력 사이의 이득 분산 가능성.</p>'
    if (RESULT / "physical_validation_probe.json").exists():
        probe = read_json(RESULT / "physical_validation_probe.json")
        text += f'<p class="caption">간단한 지도·현재 목표 외삽 기준: 현재 목표 유지 가정의 향후 5초 최소 측정 여유로 만든 단조 점수. 같은 검증 창의 AP {n(probe["5s_AP"])}. 실제 미래 위치·채널 미사용. 이 단순 기준과 학습 모델의 차이에 대한 검증 결과이며, 모든 비학습 방법의 열등성 주장 제외.</p>'
    reps = []
    prefix = choice["run"].rsplit("_s", 1)[0]
    for seed in [51, 62, 73]:
        f = RUNS / f"{prefix}_s{seed}" / "run.json"
        if f.exists():
            v = read_json(f)
            reps.append(
                [
                    seed,
                    n(v["validation_ap"]),
                    n(v["early_ap"]),
                    "대표" if seed == 51 else "반복 확인 전용",
                ]
            )
    text += (
        "<h3>동일 설정의 난수 반복 · 검증 자료</h3>"
        + table(["시드", "5초 AP", "3~4초 AP", "용도"], reps)
        + '<p class="caption">반복 시드의 최고값으로 대표 모델 교체 제외. 고정 데이터·전처리에서 학습 난수의 변동 확인. Transformer 반복의 경우 공통 B3 사전 학습 가중치 유지.</p>'
        + figure(
            fig["learning"],
            "개선 Transformer의 학습 갱신별 검증 AP. 5초 위험과 3~4초 새 사건의 서로 다른 정답.",
        )
    )
    body += section("selection", "3. 모델 선정 근거", text)
    text = table(
        ["항목", "설정"],
        [
            [
                "지도·분할",
                "실제 건물 배치·높이 기반 45지역. 학습 30·검증 7·기존 평가 8지역, 공간 집단 분리",
            ],
            [
                "고정 임무",
                "400m × 400m · 고도 50m · 전체 8대 · 수색 3대 · 중계 5대 · 명목 수색 V 대형 40m",
            ],
            [
                "비행 기록",
                "원본 270 + 추가 학습 180 + 동결 후 새 난수 확인 24 = 474비행",
            ],
            [
                "학습 입력",
                "원본 327,680개 + 추가 기록 184,320개 = 512,000개. 손실 가중치 묶음별 50%",
            ],
            [
                "검증 분리",
                "모델 선택 21비행·57,344개 창 / 확률·경고 보정 21비행·114,688개 창",
            ],
            ["실행 주기", "0.5초 관측·과거 20초·1초 추론. 성공 수신 비콘 최소 3회"],
            [
                "목표 정보",
                "현재 명목 목표 xy float64 16B + 명령 시각 4B. 기존 128B 비콘 안의 배정 가정",
            ],
            [
                "전송 범위",
                "1Hz/드론, 8개 5ms 비콘 슬롯 예약률 4%. 비트 직렬화·MAVLink 실물 송수신 미구현",
            ],
            [
                "학습 방식",
                "PC 오프라인 지도학습, 실행 시 드론별 로컬 관측 추론. 긴급 비행 중 학습 의무 없음",
            ],
        ],
    )
    text += bullets(
        [
            "이웃 목표는 실제 수신 비콘의 수신 기록에 따라서만 캐시 갱신. 소프트웨어 로그에서 재구성한 관측 계약 확장.",
            "새 난수 24개는 같은 기존 평가 지역의 채널·간섭·패킷·비콘 난수 확인. 완전히 새 도시의 맹검 검증과 구분.",
            "평가 지역은 이전 개발에서 확인한 지역. 학습·검증 지역과의 분리와 연구 단계의 사전 관찰 여부 구분.",
            "일부 건물 높이의 지역 평균 대체 표시 포함. 평탄 지면·고도 고정 가정. 실제 측정 전파 지도 재현에 대한 주장 제외.",
        ]
    )
    text += figure(fig["regions"], "지역별 전체 평가 AP. 공간 집단 차이를 구분한 비교.")
    ci = r["cluster_macro_ap_interval"]
    text += f'<p class="caption">공간 집단 {ci["clusters"]}개의 지역 평균 AP 차이: {n(ci["mean_difference"],4)} · 재표집 95% 구간 {n(ci["percentile95"][0],4)}~{n(ci["percentile95"][1],4)}. 수백만 겹치는 창을 독립 표본으로 간주하지 않는 탐색적 비교.</p>'
    text += table(
        ["간섭 조건", "기존 AP", "최종 AP", "최종 재현율"],
        [
            [
                {
                    "clear": "간섭 없음",
                    "continuous": "연속 간섭",
                    "intermittent": "구간 간섭",
                }[k],
                n(v["baseline"]["average_precision"]),
                n(v["new"]["average_precision"]),
                p(v["new"]["recall"]),
            ]
            for k, v in r["by_condition"].items()
        ],
    )
    body += section("environment", "4. 데이터와 시뮬레이터 연결", text)
    text = table(
        ["구성", "현재 구현"],
        [
            [
                "채널",
                "ITU 자유공간 손실 + 평탄 지면·건물 지붕 프로파일의 Bullington 근사",
            ],
            ["주파수·전력", "2.4GHz · 1MHz · 송신 10dBm"],
            ["시간 변화", "시간 상관 섀도잉 σ 1.5dB · 상관 시간 4초"],
            [
                "외부 간섭",
                "간섭 없음 / 연속 / 일부 구간. 위치·출력·활성 시각은 예측 입력에서 제외",
            ],
            [
                "링크 기준",
                "데이터·역방향 ACK를 고려한 양방향 홉 성공 확률 각각 0.9 이상",
            ],
            ["MAC·경로", "이상적 TDMA 5ms 슬롯 · 전역 정보 기반 기대 전송 비용 경로"],
            [
                "검증 범위",
                "3GPP TR 36.777 계수 미적용. 정밀 PHY/MAC·실물 채널 검증과 구분",
            ],
        ],
    )
    text += "<p>모형별 동일 채널·패킷·이동 기록. 링크 불가용과 군집 전체 연결 단절·패킷 전량 손실은 서로 다른 지표. 현재 개선 수치는 예측·알림 성능, 재배치 후 연결 유지율 향상과 구분.</p>"
    body += section("channel", "5. 통신 모델과 해석 범위", text)
    rt = runtime
    integrity = read_json(RESULT / "raw_integrity.json")
    text = table(
        ["검수", "결과"],
        [
            ["자동 검사", str(audit["tests_passed"]) + "개 통과"],
            ["원시 기록·지도", audit["data_summary"]],
            [
                "학습/실행 입력",
                "검증 창 128개의 입력 차이 " + n(audit["bank_max_difference"], 6),
            ],
            ["현재 목표 캐시", "실제 수신 갱신·미래 명령 배제·비정수 목표 일치 검사"],
            [
                "실행 지원 조건",
                f'{integrity["operational_support_gate_checked_records"]}개 검증·시험 비행의 {integrity["operational_support_gate_checked_windows"]:,}개 창. 일괄 평가 선별과 실제 지도·명령 관측 지원 조건의 일치',
            ],
            ["가중치·보정·정책", "평가 전 해시 고정 및 최종 출력 일치"],
            [
                "전체 재생",
                "테헤란: 학습 지역 시연. 서울03·부천중동: 평가 지역 시연. 모두 시작~종료 구간",
            ],
        ],
    )
    text += table(
        ["CPU 추론 범위", "중앙값", "95백분위"],
        [
            [
                "전체 56개 방향 링크·지도 계산 포함",
                n(rt["full_56_with_map"]["median_ms"], 2) + "ms",
                n(rt["full_56_with_map"]["p95_ms"], 2) + "ms",
            ],
            [
                "드론 1대·7개 이웃",
                n(rt["one_observer_7_links"]["median_ms"], 2) + "ms",
                n(rt["one_observer_7_links"]["p95_ms"], 2) + "ms",
            ],
        ],
    )
    text += bullets(
        [
            "현재 CPU·1스레드·초기 지도/모델 로딩 제외의 측정. 실제 비행 제어기 처리 시간 보장과 구분.",
            "후속 PPO/APF 등에 유효한 링크 위험·관측 상태 전달 가능. 학습 기반 재배치와 연결 유지율·단절 시간·PDR의 통합 검증은 후속 단계.",
            "고정 트리 위험 입력을 이용한 정책 학습 가능. 예측기 파라미터까지 공동 역전파하는 경우 신경망 참고 모델과 별도 학습 설계 필요.",
            "새 재배치 정책·대형·대수·속도 변화 시 이동 분포 변화에 대한 추가 기록 수집·재평가 필요.",
            "현재 목표는 3초 단위 발행 정보. 5초·8초 예측 전 구간의 실제 미래 이동 계획을 알고 있는 조건과 구분.",
            "장애 예방 성공 사례를 고정 경로의 예측 오경보로 단순 계산하지 않는 후속 평가 설계 필요.",
        ]
    )
    body += section("review", "6. 최종 검수와 기타 설명", text)
    text = table(
        ["용어", "정의"],
        [
            [
                "AP",
                "위험 사례를 상위로 정렬하는 품질. Σ Pₖ(Rₖ−Rₖ₋₁). 정확도 백분율과 별개",
            ],
            ["정밀도", "TP/(TP+FP). 경고 창 중 실제 미래 위험 비율"],
            ["재현율", "TP/(TP+FN). 실제 위험 창의 포착 비율"],
            ["FPR", "FP/(FP+TN). 정상 미래 창의 잘못된 경고 비율"],
            ["Brier", "예측 확률과 0/1 정답 차이의 제곱 평균. 낮을수록 우수"],
            ["첫 경고 3초 선행 포착", "새 사건 중 첫 경고가 최소 3초 앞선 사건의 비율"],
            [
                "오경보/드론·시간",
                "새 단절과 일치하지 않는 첫 알림 수 / 전체 드론 운용 시간",
            ],
            [
                "통신 성공 확률 0.9",
                "물리 링크 가용성 기준. 모델이 출력한 미래 장애 위험과 구분",
            ],
            ["관측 미지원", "확률 None과 별도 상태. 위험 0으로 대체 금지"],
        ],
    )
    refs = [
        (
            "Ke et al. (2017), LightGBM",
            "https://proceedings.neurips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf",
            "일반 GBDT 원리의 근거. 현재 구현은 행·열 표본 추출 GBDT 설정, GOSS 명시 구현 주장 제외.",
        ),
        (
            "Vaswani et al. (2017), Attention Is All You Need",
            "https://arxiv.org/abs/1706.03762",
            "자기 주의·위치 정보의 원리. 원 논문의 번역 성능을 UAV 예측 우월성 근거로 사용하지 않는 범위.",
        ),
        (
            "Yèche et al. (ICML 2023), Temporal Label Smoothing for Early Event Prediction",
            "https://proceedings.mlr.press/v202/yeche23a.html",
            "시점 지표와 사건·오경보 평가의 구분에 대한 참고. 의료 실험과 UAV 성능의 구분, TLS 자체 구현 미포함.",
        ),
        (
            "MAVLink Common Message Set",
            "https://mavlink.io/en/messages/common.html#POSITION_TARGET_LOCAL_NED",
            "현재 목표와 실제 미래 위치의 구분 참고. 본 비콘은 별도 추상 배정 계약.",
        ),
    ]
    text += (
        "<h3>근거 자료</h3><ol>"
        + "".join(
            '<li><a href="'
            + url
            + '">'
            + esc(title)
            + '</a><br><span class="muted">'
            + note
            + "</span></li>"
            for title, url, note in refs
        )
        + "</ol>"
    )
    body += section("terms", "7. 용어와 근거 자료", text)
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f7;color:#182a40;font-family:"Noto Sans KR","Malgun Gothic",sans-serif;line-height:1.75}header{background:#142b42;color:white;padding:38px max(24px,calc((100vw - 1180px)/2))}header small{color:#9ccbf4;letter-spacing:.14em}h1{font-size:31px;line-height:1.4;margin:12px 0}header p{color:#c3d5e7;margin:0;font-size:14px}nav{display:flex;flex-wrap:wrap;gap:20px;max-width:1244px;margin:auto;padding:20px 32px;font-size:13px}a{color:#1c5db0;text-decoration:none}main{max-width:1244px;margin:auto;padding:0 32px 40px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:4px 0 22px}.metric,section{background:white;border:1px solid #dbe4ee;border-radius:12px}.metric{padding:20px 17px}.metric span,.metric small{display:block;color:#64758b;font-size:12px}.metric strong{display:block;color:#1853a0;font-size:22px;margin:12px 0 7px;letter-spacing:-.035em}section{padding:27px;margin:20px 0;scroll-margin-top:20px}h2{font-size:21px;line-height:1.5;margin:0 0 18px}h3{font-size:16px;margin:25px 0 12px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}th,td{padding:12px 11px;text-align:left;vertical-align:top;border-bottom:1px solid #e4ebf3}th{background:#eef3f9;color:#355270}td:first-child{font-weight:500;min-width:115px}p,li{font-size:14px}li{margin:7px 0}figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption,.caption,.muted{font-size:12px;color:#687c92}b{color:#1853a0}@media(max-width:850px){main{padding:0 14px 30px}nav{padding:18px 14px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:20px 15px}h1{font-size:26px}.metric strong{font-size:18px}}"""
    doc = (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>통신 장애 예측 · 최종 검증</title><style>'
        + css
        + "</style></head><body><header><small>UAV COMMUNICATION / FORECAST VERIFICATION</small><h1>통신 장애 예측 · 최종 검증</h1><p>동일 조건 비교 · 선정 근거 · 사건 경고 · 독립 난수 확인 · 실행 검수</p></header><nav>"
        + "".join(
            '<a href="#' + k + '">' + v + "</a>"
            for k, v in [
                ("performance", "예측 성능"),
                ("alerts", "조기 경고"),
                ("selection", "선정 근거"),
                ("environment", "시뮬레이터"),
                ("review", "기타 설명"),
            ]
        )
        + "</nav><main>"
        + body
        + "</main></body></html>"
    )
    (ROOT / "docs/통신장애_예측성능_최종검증.html").write_text(doc, encoding="utf-8")
    write_json(
        RESULT / "report_summary.json",
        {
            "selected": choice,
            "original48": {"baseline": old, "reference": ref, "new": new},
            "first_alert5s": {"baseline": oa, "new": na},
            "new24": {"baseline": c["baseline_retuned"]["all"], "new": c["new"]["all"]},
            "control_repositioning_implemented": False,
        },
    )


if __name__ == "__main__":
    main()
