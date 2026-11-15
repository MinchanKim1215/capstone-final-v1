"""Final V1 재생 HTML에 저장된 실제 비행 기록으로 개선안의 근거를 확인한다 (시뮬레이터·지도 불필요).

  python -m final_v1_relocation.replay_audit

1. 약점 2 가설: 단절 중 판단 시점에서 V18이 insufficient_observation으로 건너뛰는 비율
   (runner가 판단 주기마다 남긴 observed = [최근 RF 수신, 비콘이 끊긴 수색기 수, 잡음, 큐])
2. 개선안 1이 새로 판단할 수 있게 되는 단절 중 판단 시점 수
3. 개선안 2: 명령 손실 14조건에서 연속 미수신 길이 (6초 시간 제한의 근거)
4. 개선안 3: 같은 V7 궤적에 기록된 V21 점수로, 문턱을 낮추면 3~7초 앞 판단 시점에서
   문턱을 넘는 단절이 몇 건 늘어나는지 (발동 기회의 상한)
"""
import base64
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import ARTIFACTS, WORKSPACE

REPLAY = WORKSPACE/"final_v1"/"Final V1 재배치 재생.html"
START_S = 21.     # 재배치 규칙이 동작하기 시작하는 시각
STEP_S = 3.
LONG_S = 10.


def load(path=REPLAY):
    text = Path(path).read_text(encoding="utf-8")
    marker = text.index('id="flight-data"')
    start = text.index(">", marker)+1
    return json.loads(text[start:text.index("</script>", start)])


def bits(encoded, count):
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return np.unpackbits(raw)[:count].astype(bool)


def blocked(observation):
    """V18 첫 조건: 최근 RF가 없거나 수색기 비콘이 하나라도 3초 넘게 끊기면 판단하지 않는다."""
    local_rf, missing = observation[0], observation[1]
    return (not local_rf) or missing > 0


def decision_ticks(run, start, stop):
    """실제로 수색망이 끊긴 상태에서 내린 판단 주기 (판단 직전 프레임이 단절)."""
    network = bits(run["connected"], int(run["duration"]*2))
    ticks = []
    for k in range(int(np.ceil(max(start, START_S)/STEP_S)), int(np.ceil(stop/STEP_S))):
        frame = int(k*STEP_S*2)
        if 0 < frame <= len(network) and not network[frame-1] and k < len(run["observed"]):
            ticks.append(k)
    return ticks


def audit_control(data, label):
    count = Counter()
    per_outage = []
    first_block = []
    for case in data["cases"]:
        if case["phase"] != "비교":
            continue
        run = next(r for r in case["runs"] if r["label"] == label)
        recoveries = [e for e in run["interventions"] if e["reason"] == "recovery"]
        count["recovery_commands"] += len(recoveries)
        count["recovery_while_down"] += sum(not e["actual_pre_action_connected"] for e in recoveries)
        for start, stop in run["cuts"]:
            if start <= 0.:
                continue
            length = stop-start
            ticks = decision_ticks(run, start, stop)
            stopped = [k for k in ticks if blocked(run["observed"][k])]
            missing = [k for k in ticks if run["observed"][k][1] > 0]
            no_rf = [k for k in ticks if not run["observed"][k][0]]
            count["outages"] += 1
            count["outage_seconds"] += length
            count["ticks_while_down"] += len(ticks)
            count["ticks_blocked"] += len(stopped)
            count["ticks_missing_search_beacon"] += len(missing)
            count["ticks_no_local_rf"] += len(no_rf)
            long = length >= LONG_S
            if long:
                count["long_outages"] += 1
                count["long_outage_seconds"] += length
                count["long_ticks_while_down"] += len(ticks)
                count["long_ticks_blocked"] += len(stopped)
                count["long_outages_all_blocked"] += bool(ticks) and len(stopped) == len(ticks)
            if ticks:
                count["outages_with_decision"] += 1
                count["outages_all_blocked"] += len(stopped) == len(ticks)
            if stopped:
                first_block.append(stopped[0]*STEP_S-start)
            per_outage.append(dict(region=case["region"], kind=case["kind"], start_s=start, length_s=length,
                                   decisions=len(ticks), blocked=len(stopped)))
    total, long_total = count["ticks_while_down"], count["long_ticks_while_down"]
    return dict(counts=dict(count),
                blocked_share=count["ticks_blocked"]/total if total else None,
                long_blocked_share=count["long_ticks_blocked"]/long_total if long_total else None,
                long_share_of_outage_time=count["long_outage_seconds"]/count["outage_seconds"]
                if count["outage_seconds"] else None,
                first_block_after_onset_s=dict(median=float(np.median(first_block)) if first_block else None,
                                               p90=float(np.percentile(first_block, 90)) if first_block else None),
                outages=per_outage)


def audit_command_loss(data):
    """중계기별 연속 미수신 길이. 강제 손실 구간(12초)과 무작위 손실을 나눈다."""
    result = {}
    for label in ("미수신 시 기본 대형", "미수신 시 자율 보정"):
        streaks = defaultdict(list)
        for case in data["cases"]:
            if case["phase"] != "명령 손실":
                continue
            run = next(r for r in case["runs"] if r["label"] == label)
            windows = run.get("stress") or []
            lost = run["lost"]
            nr = run["n"]-run["ns"]
            for relay in range(nr):
                current, began = 0, None
                for tick in range(len(lost)+1):
                    missing = tick < len(lost) and lost[tick] is not None and relay in lost[tick]
                    if missing:
                        began = tick if current == 0 else began
                        current += 1
                    elif current:
                        t = began*STEP_S
                        forced = any(a <= t < b and relay+run["ns"] == r for a, b, r in windows)
                        streaks["forced" if forced else "random"].append(current*STEP_S)
                        current = 0
        result[label] = {kind: dict(count=len(v), within_6s=int(sum(x <= 6. for x in v)),
                                    median_s=float(np.median(v)) if v else None, max_s=float(max(v)) if v else None)
                         for kind, v in streaks.items()}
    return result


CONTROL_THRESHOLD = .1775
SCALES = (1., .9, .8, .7, .6)
# 기록 점수 검증용: Final V1 경보 평가의 보정 문턱 근처. 이 값에서 RESULTS.md의
# "V21 혼합 점수 127/246, 정밀도 65.46%, 오경보·늦은 경보 3.15/편대·시간"이 재현된다.
ALERT_CHECK_THRESHOLD = .4457


def alert_score(flights, threshold, low=3., high=7.):
    """final_v1.evaluation.score_advisory와 같은 규칙 (상승 경보, 3~7초 앞 단절 하나만 적중)."""
    hits = alerts = events = 0
    hours = 0.
    for h, t, s, onsets in flights:
        hours += h
        events += len(onsets)
        active = s >= threshold
        starts = t[active & ~np.r_[False, active[:-1]]]
        alerts += len(starts)
        hit = np.zeros(len(onsets), dtype=bool)
        for when in starts:
            first, stop = np.searchsorted(onsets, [when+low, when+high])
            for k in range(first, stop):
                if not hit[k]:
                    hit[k] = True
                    break
        hits += int(hit.sum())
    return dict(hits=hits, events=events, alerts=alerts, precision=hits/alerts if alerts else None,
                false_or_late_per_h=(alerts-hits)/hours)


def audit_threshold(data):
    flights, best = [], []
    ticks, above = 0, Counter()
    for case in data["cases"]:
        if case["phase"] != "비교":
            continue
        run = next(r for r in case["runs"] if r["label"] == "기존 V7 PPO")
        s = np.array([np.nan if v is None else v for v in run["risk"]["v21"]], dtype=float)
        t = np.arange(len(s))*STEP_S
        onsets = np.array([a for a, _ in run["cuts"] if a > 0.])
        flights.append((run["duration"]/3600., t, s, onsets))
        usable = np.isfinite(s)
        ticks += int(usable.sum())
        for k in SCALES:
            above[k] += int((s[usable] >= CONTROL_THRESHOLD*k).sum())
        for onset in onsets:
            window = (t > onset-7.) & (t <= onset-3.) & usable
            best.append(float(s[window].max()) if window.any() else np.nan)
    best = np.asarray(best)
    levels = []
    for k in SCALES:
        threshold = CONTROL_THRESHOLD*k
        levels.append(dict(scale=k, threshold=threshold, events_over_threshold=int((best >= threshold).sum()),
                           decision_ticks_over_threshold=above[k], **{"alert_"+key: value for key, value in
                                                                     alert_score(flights, threshold).items()}))
    below = best[np.isfinite(best) & (best < CONTROL_THRESHOLD)]
    edges = [0., .01, .05, CONTROL_THRESHOLD*.6, CONTROL_THRESHOLD]
    return dict(events=len(best), decision_ticks=ticks, levels=levels,
                validation=alert_score(flights, ALERT_CHECK_THRESHOLD),
                below_threshold_score_bins={f"{a:.4f}-{b:.4f}": int(((below >= a) & (below < b)).sum())
                                            for a, b in zip(edges[:-1], edges[1:])},
                no_score_in_window=int(np.isnan(best).sum()))


def markdown(result):
    v21, v7 = result["v21"], result["v7"]
    c = v21["counts"]

    def row(name, item):
        k = item["counts"]
        return (f"| {name} | {k['outages']} | {k['ticks_while_down']} | {k['ticks_blocked']} "
                f"({item['blocked_share']*100:.1f}%) | {k['ticks_missing_search_beacon']} | {k['ticks_no_local_rf']} | "
                f"{k['long_ticks_blocked']}/{k['long_ticks_while_down']} ({item['long_blocked_share']*100:.1f}%) | "
                f"{k['recovery_commands']} ({k['recovery_while_down']}) |")
    lines = ["# 재생 기록으로 본 개선안 근거", "",
             "Final V1 재생 HTML의 확인 비행 42개 실제 기록입니다. runner가 3초 판단마다 남긴 "
             "`observed`(최근 RF 수신 여부, 비콘이 끊긴 수색기 수)로 V18의 첫 조건을 재현했습니다.", "",
             "## 약점 2: 단절 중 V18 판단이 멈추는가", "",
             "| 방식 | 새 단절 | 단절 중 판단 시점 | V18이 건너뛴 시점 | 수색 비콘 누락 | 최근 RF 없음 | 10초 이상 단절에서 건너뛴 시점 | 복구 명령 (실제 단절 중) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             row("V21 선제 제어", v21), row("기존 V7 PPO", v7), "",
             f"* V21 단절 중 판단 시점 {c['ticks_while_down']}개 중 {c['ticks_blocked']}개"
             f"({v21['blocked_share']*100:.1f}%)에서 V18이 `insufficient_observation`으로 판단을 건너뛰었습니다.",
             f"* 판단 시점이 있었던 단절 {c['outages_with_decision']}건 중 {c['outages_all_blocked']}건은 모든 판단 시점이 막혔습니다. "
             f"10초 이상 단절은 {c['long_outages']}건 중 {c['long_outages_all_blocked']}건이 전부 막혔습니다.",
             f"* 첫 차단은 단절 시작 뒤 중앙값 {v21['first_block_after_onset_s']['median']:.1f}초"
             f"(90% {v21['first_block_after_onset_s']['p90']:.1f}초)에 나타났습니다.",
             f"* 개선안 1은 이렇게 막힌 시점 {c['ticks_blocked']}개에서 복구 전용 판단을 실행합니다 "
             f"(그중 10초 이상 단절 {c['long_ticks_blocked']}개).", "",
             "## 개선안 2: 명령 미수신이 얼마나 이어지는가 (명령 손실 14조건)", "",
             "| 미수신 대응 | 손실 종류 | 연속 미수신 | 6초 이내 | 중앙값 | 최대 |", "|---|---|---:|---:|---:|---:|"]
    for label, kinds in result["command_loss"].items():
        for kind, s in kinds.items():
            lines.append(f"| {label} | {'강제(12초 구간)' if kind == 'forced' else '무작위'} | {s['count']} | "
                         f"{s['within_6s']} | {s['median_s']:.1f}초 | {s['max_s']:.1f}초 |")
    th = result["threshold"]
    check = th["validation"]
    base = th["levels"][0]
    low = th["levels"][-1]
    lines += ["", "## 개선안 3: 문턱을 낮추면 몇 건을 더 잡는가 (같은 V7 궤적의 V21 점수)", "",
              f"기록된 점수 검증: 경보 평가 규칙으로 {check['hits']}/{check['events']}, 정밀도 {check['precision']*100:.2f}%, "
              f"오경보·늦은 경보 {check['false_or_late_per_h']:.2f}회/시간. Final V1 RESULTS.md의 V21 혼합 점수와 같습니다.", "",
              "| 제어 문턱 | 3~7초 앞 판단에서 문턱을 넘은 단절 | 문턱을 넘은 판단 시점 | 상승 경보 제때 적중 | 경보 정밀도 | 오경보·늦은 경보/시간 |",
              "|---|---:|---:|---:|---:|---:|"]
    for level in th["levels"]:
        lines.append(f"| ×{level['scale']:.1f} ({level['threshold']:.4f}) | {level['events_over_threshold']}/{th['events']} "
                     f"({level['events_over_threshold']/th['events']*100:.1f}%) | {level['decision_ticks_over_threshold']}/"
                     f"{th['decision_ticks']} | {level['alert_hits']} | {level['alert_precision']*100:.1f}% | "
                     f"{level['alert_false_or_late_per_h']:.2f} |")
    bins = th["below_threshold_score_bins"]
    missed = sum(bins.values())
    lines += ["", f"* 현재 문턱 0.1775에서 {th['events']-base['events_over_threshold']-th['no_score_in_window']}건은 "
              f"3~7초 앞 판단 시점에서 점수가 문턱 아래였습니다. 문턱을 ×0.6까지 낮춰도 그중 "
              f"{low['events_over_threshold']-base['events_over_threshold']}건만 더 넘습니다.",
              "* 문턱 아래 단절의 최고 점수 분포: " + ", ".join(f"{k} {v}건" for k, v in bins.items()) + f" (합계 {missed}건).",
              "* 놓친 단절 대부분은 문턱 근처가 아니라 점수가 매우 낮습니다. 문턱 조정만으로 늘릴 수 있는 발동 기회는 작고, "
              "놓친 단절을 줄이려면 예측 점수 자체(입력·모델)를 개선해야 합니다.",
              "* V7 궤적 기준 상한입니다. V21 제어가 개입하면 궤적이 달라지고, 문턱을 넘어도 물리 검사에서 후보가 없으면 움직이지 않습니다.", ""]
    return "\n".join(lines)


def main():
    data = load()
    result = dict(v21=audit_control(data, "V21 선제 제어"), v7=audit_control(data, "기존 V7 PPO"),
                  command_loss=audit_command_loss(data), threshold=audit_threshold(data))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS/"replay_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = markdown(result)
    (ARTIFACTS/"REPLAY_AUDIT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
