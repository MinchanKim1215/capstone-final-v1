"""Produce the offline replay and bind the verified Final V1 release."""
from base64 import b64encode
from pathlib import Path
import json

import numpy as np

from adaptive_joint_expanded import LEGACY
from adaptive_joint_expanded.features import OFFSETS
from forecast_rebuild.contracts import verify_prior

from . import ARTIFACTS, ROOT, VERSION
from .evaluation import forecast_model, forecast_scores, key
from .protocol import maps, now, read, sha, status, write

ARM_LABELS = dict(nominal="기본 대형", reactive="단절 후 복구", v7="기존 V7 PPO", v21="V21 선제 제어",
                  physics_trigger="물리 위험으로 발동", no_short="임박 위험 검사 제외",
                  no_backup="우회 경로 가점 제외", ppo_base="추가 예측 입력 없는 PPO",
                  ppo_continuous="연속 예측 입력 PPO", neutral="미수신 시 기본 대형",
                  local="미수신 시 자율 보정", local_no_recovery="자율 보정 · 복구 분기 제외")
PATTERN_LABELS = dict(abrupt="급락형", gradual="점진형", mixed="혼합형",
                      other="기타", insufficient_history="이력 부족")
FORECAST_LABELS = dict(v20="V20 물리 점수", v21="V21 혼합 점수", joint_early="공동학습 3~7초",
                       joint_short="공동학습 0.5~3초", joint_3to5="공동학습 3~5초",
                       joint_5to7="공동학습 5~7초")


def packed(values):
    return b64encode(np.packbits(values,axis=-1).tobytes()).decode("ascii")


def geometry(registry, regions):
    output = {}
    for region in regions:
        scene = read(LEGACY/registry[region]["path"])
        shapes = []
        for b in scene["buildings"]:
            geom = b["geometry_local_m"]
            coordinates = geom["coordinates"]
            polys = [coordinates] if geom["type"] == "Polygon" else coordinates
            shapes.extend([int(b["height_m"] >= 50), poly] for poly in polys)
        output[region] = dict(name=scene["metadata"]["region_name"], shapes=shapes)
    return output


def payload(row, name, model):
    path = Path(row["archive_path"])
    if sha(path) != row["archive_sha256"]:
        raise ValueError("Replay trace changed")
    with np.load(path,allow_pickle=False) as z:
        ns, n = row["search_drones"], row["total_drones"]
        i,j = np.triu_indices(n,1)
        links = z["links"][:,i,j]
        events = []
        for e in row["intervention"]:
            show = {k:e[k] for k in ("time_s","reason","expert","changed_relay","actual_pre_action_connected",
                                     "realized_endpoint","base_action","action","command_received",
                                     "executed_as_requested") if k in e}
            show["margin"] = e.get("base_margin_db",e.get("original_margin_db"))
            show["gain"] = e.get("robust_gain_db_seconds",e.get("estimated_cost_reduction"))
            show["expected_cut"] = e.get("projected_onset_time_s")
            show["risk"] = e.get("early_score")
            show["threshold"] = e.get("threshold")
            events.append(show)
        result = dict(label=ARM_LABELS[name], ns=ns,n=n,duration=row["duration_s"], cuts=row["persistent_cut_intervals"],
                      positions=z["positions"].astype(np.int16).reshape(-1,n*2).tolist(),
                      edges=packed(links), edges_stride=(len(i)+7)//8, connected=packed(z["network"]),
                      pdr=round(row["timely_pdr"]*100,2), interventions=events)
        if "forecast_features" in z:
            scores = forecast_scores(model,z["forecast_features"])
            result["risk"] = {k:[round(float(s),4) if np.isfinite(s) else None for s in scores[k]]
                              for k in ("v20","v21","joint_early")}
        if "observed" in z:
            result["observed"] = np.round(z["observed"].astype(float),1).tolist()
        if name.startswith("ppo_"):
            base, proposed = z["original"], z["requested"]
            result["ppo_changes"] = [[[int(i), *OFFSETS[next_actions[i]].astype(int).tolist()]
                                      for i in np.flatnonzero(old_actions != next_actions)]
                                     for old_actions,next_actions in zip(base,proposed)]
        if "received" in z:
            rx = z["received"]
            result["lost"] = [np.flatnonzero(~r).tolist() for r in rx]
            result["fallback"] = [r.tolist() if (~p).any() else [] for r,p in zip(z["fallback"],rx)]
            result["stress"] = row["stress_windows"]
    return result


def render_viewer(comparison, groups, stress, model):
    cases = {}
    chosen = comparison["candidate"]
    for name in dict.fromkeys(("nominal","v7","v21","ppo_continuous",chosen)):
        for row in groups[name]:
            spec = row["spec"]
            case = cases.setdefault(("confirm",*key(row)),dict(kind=spec["kind"], phase="비교", region=spec["region"],
                                                             direction=spec["direction"], runs=[]))
            if name == chosen:
                case["preferred"] = len(case["runs"])
            case["runs"].append(payload(row,name,model))
    for name, rows in stress.items():
        for row in rows:
            spec = row["spec"]
            case = cases.setdefault(("stress",*key(row)),dict(kind=spec["kind"], phase="명령 손실",region=spec["region"],
                                                            direction=spec["direction"],runs=[]))
            case["runs"].append(payload(row,name,model))
    for event in comparison["same_history_examples"]:
        case = cases[("confirm",*key(event))]
        case["featured"] = dict(time=event["action_time_s"], onset=event["baseline_onset_s"],
                                pattern=PATTERN_LABELS[event["pattern"]])
    value = dict(cases=list(cases.values()), maps=geometry(maps(),sorted({c["region"] for c in cases.values()})))
    html = (ROOT/"viewer.html").read_text(encoding="utf-8")
    inline = json.dumps(value, ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</",r"<\/")
    if html.count("<!-- FLIGHT_DATA -->") != 1:
        raise ValueError("Missing unique replay template insertion marker")
    out = ROOT/"Final V1 재배치 재생.html"
    pending = out.with_suffix(".pending")
    pending.write_text(html.replace("<!-- FLIGHT_DATA -->",inline),encoding="utf-8")
    pending.replace(out)
    return out, len(cases)


def changed_actions(rows):
    ref = {key(r):r for r in rows["v21"]}
    result = {}
    for arm in ("physics_trigger","no_short","no_backup","ppo_base","ppo_continuous"):
        changed_commands = 0
        changed_motion = 0
        for row in rows[arm]:
            baseline = ref[key(row)]
            with np.load(baseline["archive_path"],allow_pickle=False) as a, np.load(row["archive_path"],allow_pickle=False) as b:
                changed_commands += not np.array_equal(a["commands"],b["commands"])
                changed_motion += not np.array_equal(a["nodes"],b["nodes"])
        result[arm] = dict(commands=changed_commands,motion=changed_motion,flights=len(rows[arm]))
    return result


def format_report(plan, training, comparison, transport, release):
    causes = read(ARTIFACTS/"causes.json")
    lines = ["# Final V1 연구 검증 결과", "",
             f"검증 완료 {release['created_at']} · 대표 제어 **{ARM_LABELS[release['controller']]}**", "",
             "## 1. 지도별 원인과 단절 전 패턴", "",
             f"345개 지도에서 기존 제어 비행의 반초 단위 무선 판정을 재현했다. "
             f"새 단절 {causes['new_outages']:,}건과 시작부터 연결이 끊긴 {causes['initially_down']}구간을 분리했다.", "",
             "| 유형 | 새 단절 | 건물 손실만 제거해 첫 1초 회복 |", "|---|---:|---:|"]
    for pattern, info in causes["by_pattern"].items():
        lines.append(f"| {PATTERN_LABELS[pattern]} | {info['events']:,} | {info['building_alone_repairs_first_second']:,} |")
    lines += ["", "급락은 마지막 0.5초에 통신 여유가 6dB 이상 감소한 경우다. 점진은 직전 프레임 이전의 "
              "두 구간 중앙값 차가 3dB 이상인 경우다. 둘 다 충족하면 혼합이다. 문턱을 4/2dB 및 8/4dB로 "
              "바꾼 집계도 저장했다. 통신 여유는 대칭 SINR 기준의 근사 dB값이며, 실제 연결 판정은 별도로 "
              "데이터·ACK 성공확률을 재현해 원본과 일치시켰다.", "",
              "## 2. 연속 예측 입력과 PPO 학습", "",
              f"훈련 지도 245개, 예측 입력 {training['supervised']['rows']:,}행. 다음 단절 시점을 0.5~3초, "
              "3~5초, 5~7초, 해당 구간 내 없음으로 나눴다. 계획된 보정 유지·기본 대형 복귀의 214개 통신 "
              "특징에서 만든 24차원 표현과 네 시간 점수, 관측 여부를 PPO에 전달했다. 기존 정책을 기준으로 "
              "추가 행동 점수만 학습했다. 같은 구조에서 새 입력만 제외한 정책도 별도로 학습했다.", "",
              "| 방식 | PPO 갱신 | 신규 학습 비행 | 행동 시점 | PPO→예측기 최대 기울기 |",
              "|---|---:|---:|---:|---:|"]
    for arm, runs in training["ppo"].items():
        lines.append(f"| {ARM_LABELS[arm]} | {len(runs)} | {sum(len(r['flights']) for r in runs)} | "
                     f"{sum(r['steps'] for r in runs):,} | {max(r['ppo_gradient_into_predictor_l2'] for r in runs):.5g} |")
    lines += ["", "## 3. 동일 조건에서 예측과 재배치 비교", "",
              "알려진 지도 14개의 세 간섭 조건에서 새 난수를 사용했다. 각 방식은 자기 명령으로 처음부터 끝까지 "
              "비행했다. V7에도 기존 Transformer 예측이 포함된다. 물리 발동 대조는 V21의 선행 위험 발동만 "
              "물리 점수로 바꾸며, 임박 위험 검사와 기존 후보 선택은 공통이다. 기존 고정 발동 문턱은 V21 "
              "0.1775, 물리 점수 0.60이다. 두 점수의 척도가 달라 문턱도 함께 바뀌는 운용 설정 비교다.", "",
              "| 방식 | 새 단절 | 새 단절/시간 | 단절 초/시간 | 기한 내 전달 | 평균 임무 시간 | 평균 중계 이동 | 완주 | 안전 위반 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,r in comparison["stats"].items():
        minutes = r["hours"]*60/r["flights"]
        lines.append(f"| {ARM_LABELS[arm]} | {r['starts']} | {r['starts_per_h']:.2f} | {r['cut_s_per_h']:.2f} | "
                     f"{r['timely_pdr']:.2%} | {minutes:.2f}분 | {r['mean_relay_m']/1000:.2f}km | "
                     f"{r['completed']}/{r['flights']} | {r['safety']} |")
    lines += ["", f"V21 기준 예방 명령 {release['prevention_commands']['v21']}회.",
              "", "| V21 대비 | 시간당 새 단절 차이 | 공간 그룹 95% 구간 | 이동이 달라진 비행 | 회피 규칙 개입 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ("physics_trigger","no_short","no_backup","ppo_continuous"):
        d = comparison["vs_v21"][arm]
        moves = release["action_differences_vs_v21"][arm]
        interventions = "–" if arm.startswith("ppo_") else str(release["prevention_commands"][arm])
        lines.append(f"| {ARM_LABELS[arm]} | {d['starts_per_h_delta']:+.2f} | [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] | "
                     f"{moves['motion']}/{moves['flights']} | {interventions} |")
    delta = comparison["continuous_vs_matched_ppo"]
    lines += ["", f"추가 예측 입력 PPO와 같은 구조의 입력 제외 PPO 차이 {delta['starts_per_h_delta']:+.2f}회/h "
              f"(95% [{delta['ci95'][0]:+.2f}, {delta['ci95'][1]:+.2f}]).",
              f"대표 제어는 사전 조건으로 결정했다. 같은 이력의 첫 명령 분기 후 12초 연결 유지 사례 "
              f"{len(comparison['same_history_examples'])}건. 전체 회피율과 구분한다.", "",
              "### 같은 V7 궤적에서 선행 예측", "",
              "| 점수 | 제때 탐지 | 정밀도 | 오경보·늦은 경보/편대·시간 | AP |",
              "|---|---:|---:|---:|---:|"]
    for name,r in comparison["alerts"].items():
        ap = f"{r['average_precision']:.2%}" if r["average_precision"] is not None else "계산 불가"
        lines.append(f"| {FORECAST_LABELS[name]} | {r['hits']}/{r['events']} ({r['recall']:.2%}) | "
                     f"{r['precision']:.2%} | {r['false_or_late_per_fleet_h']:.2f} | {ap} |")
    lines += ["", "문턱은 별도 보정 비행에서 오경보·늦은 경보 시간당 4회 이내로 고정했다. 각 시간 구간은 "
              "실제 단절 전체를 분모로 두며 가능한 3초 제어 시점 수가 다르다. 위 구간 비교는 경보 갱신 주기 "
              "실험과 구분한다.", "",
              "| 단절 유형 | V7 궤적 새 단절 | V20 제때 탐지 | V21 제때 탐지 | 공동학습 제때 탐지 | 대표 제어에서 새 단절 |",
              "|---|---:|---:|---:|---:|---:|"]
    for pattern,label in PATTERN_LABELS.items():
        counts = comparison["pattern_counts"]
        if not counts["v7"].get(pattern,0) and not counts[release["controller"]].get(pattern,0):
            continue
        alerts = [comparison["alerts"][v]["by_pattern"].get(pattern,{}).get("hits",0)
                  for v in ("v20","v21","joint_early")]
        lines.append(f"| {label} | {counts['v7'].get(pattern,0)} | {alerts[0]} | {alerts[1]} | {alerts[2]} | "
                     f"{counts[release['controller']].get(pattern,0)} |")
    lines += ["", "유형별 탐지는 같은 V7 사건에 매칭했다. 대표 제어의 단절 수는 별도 비행에서 발생한 사건의 "
              "집계이므로, 두 열의 차이를 개별 사건의 회피 횟수로 해석할 수 없다.", "",
              "## 4. 명령 전달, 복구와 재생 화면", "",
              "첫 중계기가 계산을 맡는 구성을 가정했다. 마지막 무선 프레임에서 얻은 홉 성공 모델로 명령 전달을 "
              "표본 추출했다. 수신 실패 시 기본 대형 또는 직전 한 주기 명령과 로컬 관측을 사용하는 규칙을 비교했다. "
              "수신 실패의 방향과 시점을 통제하는 강제 손실을 함께 사용했다.", "",
              "| 수신 실패 대응 | 원격 명령 수신 | 새 단절 | 10초 내 안정 복구 | 복구 중앙값 | 안전 위반 |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode,r in transport["transport"].items():
        median = f"{r['median_stable_recovery_s']:.1f}초" if r["median_stable_recovery_s"] is not None else "자료 없음"
        lines.append(f"| {ARM_LABELS[mode]} | {r['remote_received']:,}/{r['remote_commands']:,} | "
                     f"{transport['control'][mode]['starts']} | {r['recovered_within_10s']}/{r['outages']} | "
                     f"{median} | {transport['control'][mode]['safety']} |")
    d = transport["fallback_delta"]
    recovery = transport["transport"]["local"]
    lines += ["", f"로컬 자율 보정과 기본 대형 대응의 시간당 새 단절 차이 {d['starts_per_h_delta']:+.2f}회/h "
              f"(공간 그룹 95% [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]). "
              f"로컬 대응 비행에서 실행 직전 실제 수색망이 끊긴 복구 명령은 "
              f"{recovery['recovery_commands_while_down']}회였다."]
    lines += ["", f"명령 손실 설정에서 채택한 대응은 **{ARM_LABELS[release['transport_fallback']]}**다. "
              "재생 화면에서는 같은 지도·난수의 경로, 통신 링크, 개입 근거, 명령 미수신과 대응을 나란히 볼 수 있다.", "",
              "공동 항법 정보와 명령 채널 모델은 시뮬레이션 가정이다. 실제 지연, 채널 점유, 장비 검증은 별도다. "
              "워크스페이스에서 확인 가능한 코드를 대상으로 통합했다. 별도로 전달되지 않은 팀원 변경분은 검증 범위 밖이다.",
              "", "[실행 방법](README.md) · [재생 화면](Final%20V1%20재배치%20재생.html)", ""]
    return "\n".join(lines)


def finish(plan, training, comparison, rows, transport, stress):
    completed_path = ARTIFACTS/"completed.json"
    if completed_path.exists():
        completed = read(completed_path)
        if completed["release_sha256"] != sha(ARTIFACTS/"release.json"):
            raise ValueError("Completed release changed")
        status("all_four_stages_verified",state="completed",**completed["counts"])
        return
    verify_prior()
    for path, expected in plan["dependency_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Frozen dependency changed: "+path)
    n = sum(len(flights) for flights in rows.values())
    m = sum(len(flights) for flights in stress.values())
    if n != len(plan["confirmation"])*len(plan["arms"]) or m != len(plan["stress"])*len(plan["transport_arms"]):
        raise ValueError("Incomplete Final V1 evidence")
    model = forecast_model(training)
    viewer, cases = render_viewer(comparison, rows, stress, model)
    local, neutral = transport["control"]["local"], transport["control"]["neutral"]
    fallback_delta = transport["fallback_delta"]
    fallback = ("local" if local["safety"] == 0 and local["completed"] == neutral["flights"]
                and fallback_delta["ci95"][1] < 0. and fallback_delta["timely_pdr_pp"] >= -.25 else "neutral")
    release = dict(version=VERSION, controller=comparison["candidate"], transport_fallback=fallback,
                   scope="simulation research", hardware_validated=False, created_at=now(),
                   complete_stages=["cause_patterns","continuous_forecast_ppo","paired_ablation",
                                    "command_transport_recovery_viewer"],
                   confirmation_flights=n, stress_flights=m, replay_cases=cases,
                   action_differences_vs_v21=changed_actions(rows),
                   prevention_commands={arm:sum(e["reason"] == "prevention" for r in batch for e in r["intervention"])
                                        for arm,batch in rows.items()},
                   transport_fallback_rule="Use local fallback only if the paired upper 95% bound for new outage "
                                           "rate is below zero, all flights complete safely, and timely PDR is within "
                                           "0.25 percentage points of nominal fallback.",
                   model=training["models"].get(comparison["candidate"]),
                   evidence_sha256={str(ARTIFACTS/name):sha(ARTIFACTS/name) for name in
                                    ("causes.json","protocol.json","training.json","preflight.json",
                                     "comparison_freeze.json","comparison.json","transport.json")},
                   viewer=dict(path=str(viewer),sha256=sha(viewer)))
    report = ROOT/"RESULTS.md"
    report.write_text(format_report(plan,training,comparison,transport,release),encoding="utf-8")
    release["report_sha256"] = sha(report)
    write(ARTIFACTS/"release.json",release)
    status("4_release_ready_for_viewer_check",state="awaiting_viewer_check",
           confirmation_flights=n,stress_flights=m,viewer=str(viewer))
