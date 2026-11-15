"""V18 복구 + V21 선제 판단에 개선안 1, 3, 4를 더한 재배치 판단기.

final_v1/control.py의 choose()를 그대로 따라가되 세 곳만 바뀐다.
  (1) V18이 insufficient_observation으로 판단을 건너뛰면 단절 중 복구 판단(recovery.py)을 실행
  (3) 발동 문턱 = V21 문턱 × threshold_scale. 원래 문턱 미만인 애매한 구간에서는 작은 이동만 허용
  (4) V21이 아무것도 바꾸지 않았을 때 차폐 근거로 우회 경로 배치(bypass.py)를 시도

V21 구성 요소(extract, load_profile, make_score, short_score, GUARDS ...)는 final_v1.control
모듈 속성으로 호출 시점에 찾는다. 그래서 transformer_upgrade/bridge.py가 같은 속성을 patch해도
그대로 합쳐진다.
"""
from collections import Counter

import numpy as np

from adaptive_joint_expanded.features import OFFSETS
from final_v1 import control as v21

from .bypass import bypass
from .options import Options
from .recovery import cut_recovery


class Controller:
    """비행 1회에 하나씩 만든다. 판단 이유를 집계한다."""

    def __init__(self, options=None):
        self.options = options or Options()
        self.v18_reasons = Counter()
        self.counts = Counter()
        self.last_bypass = -np.inf
        self.previous_lost = frozenset()

    def choose(self, arm, planner, actions, logits, valid, local, phase, now, base_data=None, allow_recovery=True):
        o = self.options
        # 직전 판단 주기(3초 전)에도 소식이 끊겼던 수색기가 있는가 (cut_evidence persistent/both)
        known = np.asarray(planner.state.known[:planner.design.n_search], dtype=bool)
        lost_now = frozenset(np.flatnonzero(~known).tolist())
        persistent = bool(lost_now & self.previous_lost)
        self.previous_lost = lost_now
        original, detail = v21.v18_choose("joint_geometry", planner, actions, logits, valid)
        self.v18_reasons[detail["reason"]] += 1
        if not np.array_equal(original, actions):
            recovery = detail["reason"] == "recovery"
            if (arm != "reactive" or recovery) and (allow_recovery or not recovery):
                return original, dict(detail, time_s=now), "retained_v18"
            if recovery:
                return actions, None, "recovery_disabled"
        # 개선안 1: 수색기 소식이 끊겨 V18이 판단하지 못한 경우의 복구 전용 판단
        if o.cut_recovery and allow_recovery and detail["reason"] == "insufficient_observation":
            candidate, event, why = cut_recovery(planner, actions, logits, valid, phase, now, o, persistent)
            self.counts["cut_recovery:"+why] += 1
            if event is not None:
                return candidate, event, "added_cut_recovery"
        if arm == "reactive":
            return actions, None, "no_recovery_action"
        if not planner.state.local_rf_available or phase == 2 or not (valid.sum(axis=1) > 1).any():
            return actions, None, "missing_rf_or_no_alternative"
        base = base_data if base_data is not None else v21.extract(planner, actions, local, phase)
        x, warning, margins, _, endpoint = base
        mode, config, early, short = v21.load_profile()
        if arm == "physics_trigger" or mode != "learned":
            score, threshold = warning.shield_score, v21.V20_THRESHOLD
        else:
            score = float(v21.make_score(early, config, x.reshape(1, -1), warning.shield_score)[0])
            threshold = config["threshold"]
        # 개선안 3: 문턱을 낮춘다. 원래 문턱과 낮춘 문턱 사이가 애매한 구간이다.
        trigger = threshold*o.threshold_scale
        ambiguous = bool(trigger <= score < threshold)
        if score < trigger:
            return self._bypass_or(actions, "below_advance_threshold", planner, logits, valid, local, phase, now, base)
        if ambiguous:
            self.counts["ambiguous_band_evaluated"] += 1
        base_deficit = v21.window_deficit(margins)
        minimum = v21.GUARDS["worst_case_gain_db_seconds"]
        if margins[0, 0] < 0 or base_deficit.max() < minimum:
            return self._bypass_or(actions, "no_physical_prevention_evidence", planner, logits, valid, local,
                                   phase, now, base)
        check_short = arm != "no_short"
        base_short = v21.short_score(short, x) if check_short else None
        count = v21.onset_mask(margins).sum(axis=-1)
        worst = int(np.argmax(base_deficit))
        window = np.argsort(margins[worst, 6:15])[:3]+6
        small_only = ambiguous and o.ambiguous_small_moves
        best = None
        for candidate, relay in v21.proposal_set(actions, logits, valid):
            if relay is None:
                continue
            if small_only and np.linalg.norm(OFFSETS[candidate[relay]]-OFFSETS[actions[relay]]) > o.small_move_m+1e-6:
                continue
            alt_x, _, alt, graph, alt_endpoint = v21.extract(planner, candidate, local, phase)
            if np.array_equal(alt_endpoint, endpoint):
                continue
            if (alt[:, 1:7] < np.minimum(0., margins[:, 1:7])-v21.GUARDS["allowed_early_drop_db"]).any():
                continue
            if (v21.onset_mask(alt).sum(axis=-1) > count).any():
                continue
            cost = v21.window_deficit(alt)
            gain = float(base_deficit.max()-cost.max())
            if gain < minimum or (cost > base_deficit+v21.GUARDS["per_path_deficit_tolerance_db_seconds"]).any():
                continue
            new_short = v21.short_score(short, alt_x) if check_short else None
            if check_short and new_short > max(v21.GUARDS["short_score_floor"],
                                               base_short+v21.GUARDS["short_score_tolerance"]):
                continue
            backup = 0. if arm == "no_backup" else float(
                v21.fragile_margin(graph[worst, window], planner.design.n_search).min())
            distance = np.linalg.norm(alt_endpoint-endpoint, axis=-1).sum()
            value = gain+.10*np.clip(backup, -15., 15.)-.02*distance
            if best is None or value > best[0]:
                best = value, candidate, alt_endpoint, relay, gain, backup, new_short
        if best is None:
            return self._bypass_or(actions, "no_robust_safe_gain", planner, logits, valid, local, phase, now, base)
        _, candidate, alt_endpoint, relay, gain, backup, new_short = best
        event = dict(time_s=now, reason="prevention", expert=arm, changed_relay=planner.design.n_search+relay,
                     base_action=int(actions[relay]), action=int(candidate[relay]), early_score=float(score),
                     threshold=float(threshold), projected_onset_time_s=now+warning.onset_in_s
                     if np.isfinite(warning.onset_in_s) else None,
                     modeled_short_score_before=base_short, modeled_short_score_after=new_short,
                     robust_gain_db_seconds=gain, backup_margin_db=backup,
                     projected_fleet_endpoint=alt_endpoint.tolist(),
                     base_margin_db=np.round(margins, 2).tolist())
        if o.threshold_scale != 1.:
            event.update(trigger_threshold=float(trigger), ambiguous_band=ambiguous, small_moves_only=small_only)
            if ambiguous:
                self.counts["ambiguous_band_prevention"] += 1
        return candidate, event, "added_prevention"

    def _bypass_or(self, actions, reason, planner, logits, valid, local, phase, now, base):
        """V21이 바꾸지 않은 경우에만 개선안 4를 시도한다."""
        o = self.options
        if not o.bypass:
            return actions, None, reason
        if now-self.last_bypass < o.bypass_cooldown_s:
            self.counts["bypass:cooldown"] += 1
            return actions, None, reason
        candidate, event, why = bypass(planner, actions, logits, valid, local, phase, now, base, o,
                                       v21.extract, v21.GUARDS["allowed_early_drop_db"])
        self.counts["bypass:"+why] += 1
        if event is None:
            return actions, None, reason
        self.last_bypass = now
        return candidate, event, "added_bypass"
