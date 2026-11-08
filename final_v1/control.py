"""Reproduce V21 and expose its experimental decision switches."""
import numpy as np

from forecast_avoidance.controller import choose as v18_choose, fragile_margin, onset_mask, proposal_set
from forecast_learning.control import GUARDS, load_profile, short_score, window_deficit
from forecast_learning.features import extract
from forecast_learning.protocol import V20_THRESHOLD
from forecast_learning.training import make_score


def choose(arm, planner, actions, logits, valid, local, phase, now, base_data=None, allow_recovery=True):
    original, detail = v18_choose("joint_geometry", planner, actions, logits, valid)
    if not np.array_equal(original, actions):
        recovery = detail["reason"] == "recovery"
        if (arm != "reactive" or recovery) and (allow_recovery or not recovery):
            return original, dict(detail, time_s=now), "retained_v18"
        if recovery:
            return actions, None, "recovery_disabled"
    if arm == "reactive":
        return actions, None, "no_recovery_action"
    if not planner.state.local_rf_available or phase == 2 or not (valid.sum(axis=1)>1).any():
        return actions, None, "missing_rf_or_no_alternative"
    x, warning, margins, _, endpoint = base_data if base_data is not None else extract(planner, actions, local, phase)
    mode, config, early, short = load_profile()
    if arm == "physics_trigger" or mode != "learned":
        score, threshold = warning.shield_score, V20_THRESHOLD
    else:
        score = float(make_score(early, config, x.reshape(1, -1), warning.shield_score)[0])
        threshold = config["threshold"]
    if score < threshold:
        return actions, None, "below_advance_threshold"
    base_deficit = window_deficit(margins)
    minimum = GUARDS["worst_case_gain_db_seconds"]
    if margins[0, 0] < 0 or base_deficit.max() < minimum:
        return actions, None, "no_physical_prevention_evidence"
    check_short = arm != "no_short"
    base_short = short_score(short, x) if check_short else None
    count = onset_mask(margins).sum(axis=-1)
    worst = int(np.argmax(base_deficit))
    window = np.argsort(margins[worst, 6:15])[:3]+6
    best = None
    for candidate, relay in proposal_set(actions, logits, valid):
        if relay is None:
            continue
        alt_x, _, alt, graph, alt_endpoint = extract(planner, candidate, local, phase)
        if np.array_equal(alt_endpoint, endpoint):
            continue
        if (alt[:, 1:7] < np.minimum(0., margins[:, 1:7])-GUARDS["allowed_early_drop_db"]).any():
            continue
        if (onset_mask(alt).sum(axis=-1) > count).any():
            continue
        cost = window_deficit(alt)
        gain = float(base_deficit.max()-cost.max())
        if gain < minimum or (cost > base_deficit+GUARDS["per_path_deficit_tolerance_db_seconds"]).any():
            continue
        new_short = short_score(short, alt_x) if check_short else None
        if check_short and new_short > max(GUARDS["short_score_floor"], base_short+GUARDS["short_score_tolerance"]):
            continue
        backup = 0. if arm == "no_backup" else float(fragile_margin(graph[worst, window], planner.design.n_search).min())
        distance = np.linalg.norm(alt_endpoint-endpoint, axis=-1).sum()
        value = gain+.10*np.clip(backup, -15., 15.)-.02*distance
        if best is None or value > best[0]:
            best = value, candidate, alt_endpoint, relay, gain, backup, new_short
    if best is None:
        return actions, None, "no_robust_safe_gain"
    _, candidate, alt_endpoint, relay, gain, backup, new_short = best
    event = dict(time_s=now, reason="prevention", expert=arm, changed_relay=planner.design.n_search+relay,
                 base_action=int(actions[relay]), action=int(candidate[relay]), early_score=float(score),
                 threshold=float(threshold), projected_onset_time_s=now+warning.onset_in_s
                 if np.isfinite(warning.onset_in_s) else None,
                 modeled_short_score_before=base_short, modeled_short_score_after=new_short,
                 robust_gain_db_seconds=gain, backup_margin_db=backup,
                 projected_fleet_endpoint=alt_endpoint.tolist(),
                 base_margin_db=np.round(margins, 2).tolist())
    return candidate, event, "added_prevention"
