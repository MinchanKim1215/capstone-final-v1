"""results.json → 한글 RESULTS.md (개선안 문서의 표 형식)."""

LABELS = {
    "v21": "V21 선제 제어 (적용 전)",
    "r1_cut_recovery": "개선안 1 · 단절 중 복구",
    "r1_model": "개선안 1 · 복구 근거: 모델",
    "r1_persistent": "개선안 1 · 복구 근거: 지속",
    "r1_both": "개선안 1 · 복구 근거: 모델+지속",
    "r3_t090": "개선안 3 · 문턱 ×0.9",
    "r3_t080": "개선안 3 · 문턱 ×0.8",
    "r3_t070": "개선안 3 · 문턱 ×0.7",
    "r3_t060": "개선안 3 · 문턱 ×0.6",
    "r3_t070_small": "개선안 3 · 문턱 ×0.7 + 애매 구간 작은 이동",
    "r4_bypass": "개선안 4 · 우회 경로 확보",
    "r13_combined": "개선안 1+3 · 복구 + 문턱 ×0.8 작은 이동",
}
FALLBACK_LABELS = {"neutral": "기본 대형 복귀", "hold": "직전 명령 유지", "reconnect": "마지막 수신 위치 복귀",
                   "local": "자율 보정 (Final V1)"}


def f(value, digits=2, suffix=""):
    if value is None:
        return "–"
    return f"{value:.{digits}f}{suffix}"


def ci(values):
    return f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def label(key):
    if "__" in key:
        arm, fallback = key.split("__")
        return f"{LABELS.get(arm, arm)} / {FALLBACK_LABELS.get(fallback, fallback)}"
    return LABELS.get(key, key)


def markdown(result):
    lines = ["# Final V1 재배치 개선안 비교 결과", "",
             f"제어 기준 `{result['controller']}` · 시작 {result['started_at']} · 종료 {result['finished_at']}", ""]
    if result.get("steps") or result.get("limit"):
        lines += [f"> 동작 확인 실행입니다 (steps={result.get('steps')}, limit={result.get('limit')}). "
                  "성능 비교에 쓰지 마세요.", ""]
    confirm = result.get("confirmation")
    if confirm:
        stats = confirm["stats"]
        order = [k for k in LABELS if k in stats]+[k for k in stats if k not in LABELS]
        lines += ["## 1. 확인 비행 (지도 14 × 간섭 3)", "",
                  "| 방식 | 새 단절/시간 | 단절 초/시간 | 10초 이상 단절 | 긴 단절 시간 비중 | 단절 길이 중앙값·90% "
                  "| 복구 명령 (단절 중 복구) | 예방 명령 | 우회 배치 | 중계 이동 | 기한 내 전달 | 지연 p95 | 완주 | 안전 위반 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in order:
            s = stats[arm]
            lines.append(
                f"| {label(arm)} | {f(s['starts_per_h'])} | {f(s['cut_s_per_h'])} | {s['long_outages']} | "
                f"{f(s['long_share_of_cut_time']*100, 1, '%')} | {f(s['outage_median_s'], 1)}·{f(s['outage_p90_s'], 1)}초 | "
                f"{s['recovery_commands']} ({s['cut_recovery_commands']}) | {s['prevention_commands']} | "
                f"{s['bypass_commands']} | {f(s['mean_relay_km'])}km | {f(s['timely_pdr']*100, 2, '%')} | "
                f"{f(s['delay_p95_mean_s'], 3)}초 | {s['completed']}/{s['flights']} | {s['safety']} |")
        lines += ["", "### V21 대비 (같은 지도·난수 쌍, 공간 그룹 부트스트랩 95% 구간)", "",
                  "| 방식 | 새 단절/시간 차이 | 95% 구간 | 10초 이상 단절/시간 차이 | 95% 구간 | 단절 초/시간 차이 "
                  "| 기한 내 전달 | 임무 시간 | 중계 이동 | 주지표 | 채택 조건 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for arm in order:
            if arm not in confirm["vs_v21"]:
                continue
            c = confirm["vs_v21"][arm]
            lines.append(
                f"| {label(arm)} | {c['starts_per_h_delta']:+.2f} | {ci(c['starts_per_h_ci95'])} | "
                f"{c['long_per_h_delta']:+.2f} | {ci(c['long_per_h_ci95'])} | {c['cut_s_per_h_delta']:+.2f} | "
                f"{f(c['timely_pdr_pp'], 2, '%p')} | {c['duration_fraction']*100:+.2f}% | {c['relay_km_delta']:+.2f}km | "
                f"{confirm['primary'][arm]} | {'충족' if confirm['adoptable'][arm] else '미충족'} |")
        lines += ["", "채택 조건은 Final V1 승격 조건과 같습니다: 안전 위반 0, 전원 완주, 기한 내 전달 손실 0.25%p 이하, "
                  "임무 시간 증가 2% 이하, 주지표 차이의 95% 구간 상한 < 0.", ""]
        lines += ["### V18 내부 판단 이유 (약점 2 확인용)", "", "| 방식 | 판단 수 | insufficient_observation 비율 | 상위 이유 |",
                  "|---|---:|---:|---|"]
        for arm in order:
            reasons = stats[arm]["v18_reasons"]
            total = sum(reasons.values())
            top = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:4])
            lines.append(f"| {label(arm)} | {total} | {f((stats[arm]['v18_insufficient_observation_share'] or 0)*100, 1, '%')} | {top} |")
        if "v21_matches_final_v1" in result:
            check = result["v21_matches_final_v1"]
            lines += ["", f"기준 재현: 모든 개선안을 끈 v21 비행 {check['identical']}/{check['checked']}개가 Final V1에 저장된 "
                      "V21 비행과 단절 구간까지 같습니다."]
        patterns = result.get("pattern_counts")
        if patterns and "error" not in patterns:
            lines += ["", "### 단절 유형별 새 단절 수", "", "| 방식 | 급락형 | 혼합형 | 점진형 | 기타 |", "|---|---:|---:|---:|---:|"]
            for arm, counts in patterns.items():
                lines.append(f"| {label(arm)} | {counts.get('abrupt', counts.get('급락형', 0))} | "
                             f"{counts.get('mixed', counts.get('혼합형', 0))} | {counts.get('gradual', counts.get('점진형', 0))} | "
                             f"{sum(v for k, v in counts.items() if k not in ('abrupt', 'mixed', 'gradual', '급락형', '혼합형', '점진형'))} |")
        elif patterns:
            lines += ["", f"단절 유형 집계 생략: {patterns['error']}"]
        lines.append("")
    stress = result.get("command_loss")
    if stress:
        lines += [f"## 2. 명령 손실 {stress['flights']}조건", "",
                  "| 재배치 / 미수신 대응 | 새 단절 | 10초 안 안정 복구 | 복구 중앙값 | 10초 이상 단절 | 안전 위반 | 원격 명령 수신 | 대응 사유 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for key, s in stress["stats"].items():
            fb = s["fallback"]
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(fb.items()) if k not in ("received", "remote_commands", "remote_received"))
            lines.append(f"| {label(key)} | {s['outages']} | {s['recovered_within_10s']}/{s['outages']} | "
                         f"{f(s['median_stable_recovery_s'], 1)}초 | {s['long_outages']} | {s['safety']} | "
                         f"{fb.get('remote_received', 0)}/{fb.get('remote_commands', 0)} | {reasons} |")
        lines += ["", "| 기본 대형 복귀 대비 | 새 단절/시간 차이 | 95% 구간 | 10초 이상 단절/시간 차이 | 95% 구간 |",
                  "|---|---:|---:|---:|---:|"]
        for key, c in stress["vs_neutral"].items():
            lines.append(f"| {label(key)} | {c['starts_per_h_delta']:+.2f} | {ci(c['starts_per_h_ci95'])} | "
                         f"{c['long_per_h_delta']:+.2f} | {ci(c['long_per_h_ci95'])} |")
        lines.append("")
    return "\n".join(lines)


def dev_markdown(result):
    stats, versus = result["stats"], result["vs_v21"]
    order = [k for k in LABELS if k in stats]
    lines = ["# 불필요한 복구 이동 줄이기 · 개발 비행 결과", "",
             f"학습 지도 {len(result['maps'])}곳({', '.join(result['maps'])}) × 간섭 3 × 난수 {result['seeds']} = "
             f"{result['flights']}개 조건. 확인 비행 지도와 겹치지 않는다.", "",
             "| 방식 | 새 단절/시간 | 단절 초/시간 | 10초 이상 단절 | 10초 이상/시간 | 단절 중 복구 명령 (실제 단절 중) | 중계 이동 | 기한 내 전달 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in order:
        st = stats[arm]
        lines.append(f"| {label(arm)} | {f(st['starts_per_h'])} | {f(st['cut_s_per_h'])} | {st['long_outages']} | "
                     f"{f(st['long_outages_per_h'])} | {st['cut_recovery_commands']} ({st['cut_recovery_while_down']}) | "
                     f"{f(st['mean_relay_km'])}km | {f(st['timely_pdr']*100, 2, '%')} |")
    lines += ["", "| V21 대비 | 새 단절/시간 | 95% 구간 | 10초 이상/시간 | 95% 구간 | 단절 초/시간 | 95% 구간 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in order:
        if arm in versus:
            c = versus[arm]
            lines.append(f"| {label(arm)} | {c['starts_per_h_delta']:+.2f} | {ci(c['starts_per_h_ci95'])} | "
                         f"{c['long_per_h_delta']:+.2f} | {ci(c['long_per_h_ci95'])} | {c['cut_s_per_h_delta']:+.2f} | "
                         f"{ci(c['cut_s_per_h_ci95'])} |")
    sel = result["selection"]
    lines += ["", f"선택 규칙(비행 전에 고정): {sel['rule']}.", "",
              f"**선택: {label(sel['selected'])}** · 조건을 만족한 설정: "
              + (", ".join(label(a) for a in sel["eligible"]) or "없음 (전체 중 선택)"), ""]
    return "\n".join(lines)
