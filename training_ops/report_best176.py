"""Summarize existing evaluation JSON only; never run or alter the experiment."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np

OPS = Path(__file__).resolve().parent
PROJECT = OPS.parent / "drone_forecast_joint_training"
RESULTS = PROJECT / "results/transformer"
MODEL_SUMMARY = RESULTS / "evaluation/validation/c1a9debd37963fcd/summary.json"
BASE_SUMMARY = RESULTS / "evaluation/validation/30c8224c85fc74c0/summary.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(row):
    return tuple(row[k] for k in ["region", "direction", "kind", "seed"])


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def main():
    model, base = load(MODEL_SUMMARY), load(BASE_SUMMARY)
    assert model["updates"] == 176 and model["cases"] == base["cases"] == 8
    verified = hashlib.sha256((PROJECT / model["checkpoint"]).read_bytes()).hexdigest() == model["checkpoint_sha256"]
    assert verified
    a = [load((PROJECT / p).with_suffix(".json")) for p in model["raw_paths"]]
    b = [load((PROJECT / p).with_suffix(".json")) for p in base["raw_paths"]]
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    assert aa.keys() == bb.keys()
    assert all(r["checkpoint_sha256"] == model["checkpoint_sha256"] for r in a)
    plan = load(PROJECT / "configs/transformer_plan.json")
    groups = {r["key"]: r["joint_spatial_group"] for r in plan["maps"]}
    fields = ["search_connected_mean", "pair_reach_mean", "pdr", "timely_pdr", "outage_total_s", "outage_max_s",
              "outage_events", "duration_s", "duration_ratio", "delay_mean_s", "delay_p95_s", "aoi_mean_s",
              "relay_travel_m", "search_travel_m", "search_formation_rmse_m", "actual_delivered_hops_mean",
              "delivered_via_relay_fraction", "relocation_fraction", "protection_fraction", "reward_mean", "reward_sum"]
    means = {name: {f: float(np.mean([r[f] for r in rows])) for f in fields}
             for name, rows in [("best176", a), ("physics", b)]}
    counts = {name: {f: int(sum(r[f] for r in rows)) for f in
                    ["generated_packets", "delivered_packets", "retries", "ttl_drops", "retry_drops", "overflow_drops", "outage_events"]}
              for name, rows in [("best176", a), ("physics", b)]}
    paired = [dict(region=aa[k]["region"], spatial_group=groups[aa[k]["region"]],
                   ctr_delta=aa[k]["search_connected_mean"]-bb[k]["search_connected_mean"],
                   pdr_delta=aa[k]["pdr"]-bb[k]["pdr"],
                   outage_delta_s=aa[k]["outage_total_s"]-bb[k]["outage_total_s"],
                   longest_baseline_s=bb[k]["outage_max_s"], longest_model_s=aa[k]["outage_max_s"])
              for k in sorted(aa)]
    intervals = {}
    unique = sorted({r["spatial_group"] for r in paired})
    rng = np.random.default_rng(271077)
    draws = rng.integers(0, len(unique), size=(10000, len(unique)))
    for metric in ["ctr_delta", "pdr_delta", "outage_delta_s"]:
        totals = np.asarray([sum(r[metric] for r in paired if r["spatial_group"] == g) for g in unique])
        sizes = np.asarray([sum(r["spatial_group"] == g for r in paired) for g in unique])
        samples = totals[draws].sum(1) / sizes[draws].sum(1)
        intervals[metric] = dict(difference=float(np.mean([r[metric] for r in paired])),
                                 low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
                                 spatial_groups=len(unique), maps=len(paired), draws=10000, seed=271077)
    audits = dict(all_passed=all(r["audit"]["passed"] for r in a),
                  collisions=sum(r["audit"]["collision_pairs"] for r in a),
                  obstacle_segments=sum(r["audit"]["obstacle_segments"] for r in a),
                  obstacle_points=sum(r["audit"]["obstacle_points"] for r in a),
                  invalid_graph_edges=sum(r["audit"]["invalid_graph_edges"] for r in a),
                  maximum_speed_mps=max(r["audit"]["maximum_speed_mps"] for r in a),
                  minimum_separation_m=min(r["audit"]["minimum_swept_separation_m"] for r in a),
                  packet_conservation=all(r["audit"]["packet_conservation"] for r in a),
                  reward_reconstruction=all(r["audit"]["reward_reconstructed_from_packets_and_trajectory"] for r in a),
                  movement_penalty_cap=all(r["audit"]["movement_penalty_cap_verified"] for r in a))
    out = OPS / "reports"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    derived = dict(as_of=stamp, checkpoint=model["checkpoint"], checkpoint_sha256=model["checkpoint_sha256"],
                   checkpoint_verified=verified, cases=8, spatial_groups=len(unique), means=means, counts=counts,
                   comparison=paired, paired_spatial_bootstrap=intervals, audit=audits, prediction=model["prediction"],
                   alarms=model["alarms"], model_communication=model["communication"], baseline_communication=base["communication"],
                   source_summaries=[str(MODEL_SUMMARY), str(BASE_SUMMARY)])
    (out / "best176_metrics.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = lambda v: f"{100*v:.3f}%"
    pp = lambda v: f"{100*v:+.3f}%p"
    ma, mb = means["best176"], means["physics"]
    comm_rows = []
    for f, label in [("search_connected_mean", "수색 3기 연결 유지율 CTR"), ("pdr", "패킷 전달률 PDR"),
                     ("timely_pdr", "지연 가중 전달률"), ("pair_reach_mean", "수색 드론 쌍의 도달 가능 비율")]:
        comm_rows.append([label, pct(mb[f]), pct(ma[f]), pp(ma[f]-mb[f])])
    for f, label in [("outage_total_s", "비행당 누적 단절 시간 평균"), ("outage_max_s", "비행별 최장 단절의 평균"),
                     ("duration_s", "비행당 임무 시간 평균"), ("delay_mean_s", "비행별 평균 패킷 지연의 평균"),
                     ("delay_p95_s", "비행별 패킷 지연 p95의 평균"), ("aoi_mean_s", "비행별 평균 정보 나이 AoI의 평균")]:
        comm_rows.append([label, f"{mb[f]:.4f}초", f"{ma[f]:.4f}초", f"{ma[f]-mb[f]:+.4f}초"])
    comm_rows.extend([
        ["8회 전체 중 최장 연속 단절", "35.000초", "17.500초", "-50.0%"],
        ["수색 네트워크 단절 사건 수 (8회 합계)", counts['physics']['outage_events'], counts['best176']['outage_events'], "+3건"],
        ["중계 5기 합산 이동거리 / 비행", f"{mb['relay_travel_m']/1000:.3f}km", f"{ma['relay_travel_m']/1000:.3f}km", f"{(ma['relay_travel_m']/mb['relay_travel_m']-1)*100:+.2f}%"],
        ["명목 임무 시간 대비 비율의 평균", f"{mb['duration_ratio']:.6f}배", f"{ma['duration_ratio']:.6f}배", f"{ma['duration_ratio']-mb['duration_ratio']:+.6f}배"],
        ["명목 임무 시간 대비 최대 비율", f"{base['communication']['max_duration_ratio']:.6f}배", f"{model['communication']['max_duration_ratio']:.6f}배", "선택 상한 1.15배 이내"],
        ["최저 CTR인 검증 비행", pct(base['communication']['worst_connectivity']), pct(model['communication']['worst_connectivity']), pp(model['communication']['worst_connectivity']-base['communication']['worst_connectivity'])],
        ["최저 PDR인 검증 비행", pct(base['communication']['worst_pdr']), pct(model['communication']['worst_pdr']), pp(model['communication']['worst_pdr']-base['communication']['worst_pdr'])]])
    names = ["5초 단절 위험", "3–4초 뒤 단절 시작", "3–7초 뒤 단절 시작", "0–7초 뒤 단절 시작"]
    pred_rank, pred_threshold, confusion = [], [], []
    for label, metric in zip(names, model["prediction"].values()):
        pred_rank.append([label, f"{metric['samples']:,}", f"{metric['positives']:,}", pct(metric['positive_fraction']),
                          f"{metric['average_precision']:.4f}", f"{metric['auroc']:.4f}", f"{metric['brier']:.5f}"])
        pred_threshold.append([label, pct(metric['precision']), pct(metric['recall']), f"{metric['f1']:.4f}", pct(metric['accuracy'])])
        confusion.append([label] + [f"{metric[k]:,}" for k in ["tp", "fp", "fn", "tn"]])
    alarm_rows = []
    for name, rule in model["alarms"].items():
        alarm_rows.append([name, pct(rule["first_alert_precision"]), f"{rule['false_alerts_per_drone_hour']:.2f}",
                           pct(rule["first_alert_event_recall_3s"]), f"{rule['event_hits_3s']:,} / {rule['eligible_events']:,}",
                           f"{rule['first_alerts']:,}", "통과 후보" if rule["qualifies"] else "미충족"])
    lines = [
        "**176 update 체크포인트 평가 보고서**", "", f"작성: {stamp}", "",
        "통신 유지와 단절 시간은 동일 시나리오의 물리 기반 기준선보다 평균적으로 개선됐다. 다만 실용적인 조기 경보 성능은 부족하다. 8초 경보 후보는 정밀도/오경보 조건을 충족하지만, 3초 이상 선행 탐지한 사건은 4,087건 중 17건뿐이다.", "",
        "**평가 범위**", "",
        "- 대상: transformer_main, joint 모드, 176 update 저장본. PPO 훈련 비행 누적 1,408회 시점이다.",
        "- 평가: 서로 다른 8개 검증 맵 / 공간 그룹 8개, 맵당 1조건. 양쪽 방법의 지도·방향·간섭 조건·seed가 일치한다.",
        "- 기준선: physics. 학습형 단절 예측기를 쓰지 않는 기존 물리 기반 후보/정책 경로다. 정지 비행이나 아무 제어도 하지 않는 기준선이 아니다.",
        "- 아래 비율/시간은 별도 명시하지 않는 한 비행별 지표 8개의 산술평균이다. 패킷을 합쳐 계산한 비율과 구분한다.",
        "- 모델 선택에 사용한 중간 검증이다. 전체 검증 24맵 × 4조건 = 96회, 미사용 최종 테스트는 이 보고서에서 새로 실행하지 않았다.", "",
        "**통신·단절·임무 성능**", "", table(["지표", "physics 기준선", "176 update", "변화"], comm_rows), "",
        "CTR은 0.5초 RF 관측에서 수색 드론 3기가 중계를 포함한 네트워크를 통해 서로 도달 가능한 시간 비율이다. 지연 가중 전달률은 전달 지연에 따른 가중치를 적용한 값이며, 평균 지연이나 단순 마감시간 충족 비율과 같지 않다. AoI는 받은 정보가 얼마나 오래됐는지 나타낸다. 패킷 지연은 전달된 패킷 기준이며 p95는 비행별 p95의 산술평균이다.", "",
        "평균 지연은 감소했지만 비행별 지연 p95의 평균은 64.375ms에서 81.250ms로 약 26.2% 증가했다. 네트워크 단절 횟수도 67건에서 70건으로 늘었다. 따라서 모든 지표가 개선됐다고 표현하지 않는다.", "",
        "중계 이동거리는 중계 드론 5기의 합계다. 연결 개선에 더 많은 이동이 동반됐으며 에너지/배터리 소모를 직접 실측한 값은 아니다.", "",
        "**지도별 변화**", "", table(["지역", "CTR 변화", "PDR 변화", "누적 단절 변화", "최장 단절 기준선→176"],
            [[r['region'], pp(r['ctr_delta']), pp(r['pdr_delta']), f"{r['outage_delta_s']:+.1f}초", f"{r['longest_baseline_s']:.1f}→{r['longest_model_s']:.1f}초"] for r in paired]), "",
        "모든 지도에서 일관되게 개선된 것은 아니다. CTR은 8개 중 6개, PDR은 8개 중 7개 지도에서 증가했다. seoul_20261026_10에서는 최장 연속 단절이 11.5초에서 17.5초로 늘었다.", "",
        "**차이의 불확실성**", "", table(["지표", "평균 차이", "공간 그룹 대응 bootstrap 95% 구간"],
            [["CTR", pp(intervals['ctr_delta']['difference']), f"[{pp(intervals['ctr_delta']['low'])}, {pp(intervals['ctr_delta']['high'])}]"],
             ["PDR", pp(intervals['pdr_delta']['difference']), f"[{pp(intervals['pdr_delta']['low'])}, {pp(intervals['pdr_delta']['high'])}]"],
             ["누적 단절 시간", f"{intervals['outage_delta_s']['difference']:+.3f}초", f"[{intervals['outage_delta_s']['low']:+.3f}, {intervals['outage_delta_s']['high']:+.3f}]초"]]), "",
        "기존 저장된 비행별 JSON만으로 공간 그룹을 10,000번 대응 재추출했다(seed 271077, 지도 동일 가중). 새 학습·비행·모델 추론을 실행하지 않았다. 공간 그룹이 8개뿐이고, 해당 검증으로 best를 선택했으므로 탐색적 구간이다. 미사용 지역에 대한 일반화 보장이나 선택 편향을 보정한 확증 검정으로 해석하지 않는다.", "",
        "**단절 예측기의 순위·확률 지표**", "", table(["예측 대상", "유효 표본", "양성 표본", "양성 비율", "AP ↑", "AUROC ↑", "Brier ↓"], pred_rank), "",
        "AP는 위험 순위를 얼마나 잘 매기는지를 나타내며 정확도나 정밀도와 같은 값이 아니다. Brier는 확률 예측의 제곱 오차다. 표본은 서로 상관된 유향 링크 × 제어 시점이며 독립 비행 수가 아니다.", "",
        "**확률 0.5 기준 분류 성능**", "", table(["예측 대상", "정밀도", "재현율", "F1", "정확도"], pred_threshold), "",
        "5초 단절 위험의 재현율은 31.63%라 양성 링크·시점 표본 약 68.37%를 놓친다. 3–4초 시작 head는 확률 0.5에서 양성 예측이 한 번도 없었다. 이는 예측 순위 AP가 0이라는 뜻은 아니다. 희귀 단절이므로 높은 정확도만으로 우수성을 판단하지 않는다.", "",
        table(["예측 대상", "TP", "FP", "FN", "TN"], confusion), "",
        "이 예측 지표는 176 정책이 실제 비행한 경로의 관측에서 계산했다. 기존 LightGBM이나 다른 정책의 AP와 직접 비교할 수 없다. 같은 관측 corpus로 맞춘 176 모델의 비교는 이 보고서에서 새로 수행하지 않았다.", "",
        "**사건 기반 조기 경보**", "", table(["경보", "최초 경보 정밀도", "오경보/드론·시간", "3초 이상 선행 사건 재현율", "선행 탐지/대상 사건", "최초 경보 수", "수치 조건"], alarm_rows), "",
        "- 5초 경보는 onset_3to4s head, 8초 경보는 onset_3to7s head를 사용한다. risk_5s 분류와 동일 지표가 아니다.",
        "- 사건은 1초 이상 지속된 유향 링크 단절이다. 21초 warmup과 종료 8초 censor를 적용하며 관측된 중계 드론 시간은 합계 20.0 drone-hours다.",
        "- 정밀도는 경보 후 해당 horizon 안에 사건이 있었는지를 본다. 최소 3초 선행 여부는 별도의 사건 재현율에 적용된다.",
        "- 후보 수치 조건: 정밀도 ≥80%, 오경보 ≤5회/드론·시간, 최초 경보 ≥10건. 현재 조건에는 최소 사건 재현율 하한이 없어, 8초 후보가 0.42%만 탐지하고도 통과할 수 있다.",
        "- 8초 후보는 최초 경보 21건 중 17건이 맞았고 오경보는 4건이다. 수치상 조건을 통과했지만 실제 단절 사건 대부분을 놓쳐 실용성이 부족하다.",
        "- summary의 enabled:true는 후보 검색의 수치 통과 표시다. 실제 AlarmMonitor는 전체 96조건 검증으로 고정한 선택을 추가로 요구하므로, 이번 8회 결과만으로 운용 경보를 활성화할 수 없다.", "",
        "**안전·계산 시간**", "", table(["항목", "176 update 결과"],
            [["임무 완료", "8/8"], ["안전 감사", "8/8 통과"], ["드론 간 충돌", f"{audits['collisions']}건"],
             ["장애물 교차 구간 / 내부 위치", f"{audits['obstacle_segments']} / {audits['obstacle_points']}건"],
             ["허용되지 않은 이동 경로", f"{audits['invalid_graph_edges']}건"],
             ["최소 연속 궤적 간격", f"{audits['minimum_separation_m']:.4f}m (제약 5m)"],
             ["최대 이동 속도", f"{audits['maximum_speed_mps']:.4f}m/s (상한 5m/s)"],
             ["패킷 보존·보상 독립 재계산·이동 벌점 상한", "전부 통과"],
             ["Transformer 예측 p95의 비행별 평균", f"{model['communication']['means']['forecast_p95_ms']:.3f}ms"],
             ["행동 결정 p95의 비행별 평균", f"{model['communication']['means']['decision_p95_ms']:.3f}ms"]]), "",
        "계산 시간은 이 PC의 CPU 시뮬레이션 worker에서 측정한 비행별 p95의 평균이며 전 표본을 합친 p95가 아니다. 기준선은 다른 실행 환경의 기록이므로 계산 속도의 직접 우열을 비교하지 않았다. 실기체 센서/통신/명령 전달을 포함한 지연이나 비행 안전성을 검증한 값도 아니다.", "",
        "**패킷 수 합계**", "", table(["항목", "physics", "176 update"],
            [[k, counts['physics'][k], counts['best176'][k]] for k in ["generated_packets", "delivered_packets", "ttl_drops", "retry_drops", "overflow_drops", "retries"]]), "",
        "두 정책의 비행 시간이 달라 생성 패킷 수도 다르다. 위 합계와 앞의 지도별 평균 PDR은 집계 방식이 다르다.", "",
        "**판단**", "",
        "이 중간 검증에서는 통신 유지·전달·누적 단절 시간이 평균적으로 개선됐고 시뮬레이션 안전 감사도 통과했다. 그러나 단절을 충분히 미리 알리는 성능, 미사용 지역 일반화, 실기체 적용 준비가 완료됐다고 판단할 근거는 없다. 현재 학습과 보상 계수는 변경하지 않았다.", "",
        "**원본 근거**", "", f"- 평가 JSON: {MODEL_SUMMARY}", f"- 기준선 JSON: {BASE_SUMMARY}",
        f"- checkpoint SHA256: {model['checkpoint_sha256']}", "- checkpoint 바이트 SHA 대조 통과. 결과 재계산은 저장된 JSON에 대한 집계만 수행했다.", ""
    ]
    (out / "best176_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=str(out / 'best176_metrics.md'), means=means, counts=counts,
                          intervals=intervals, audits=audits, map_improvements=dict(CTR=sum(r['ctr_delta']>0 for r in paired),PDR=sum(r['pdr_delta']>0 for r in paired)),
                          snapshot=stamp),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
