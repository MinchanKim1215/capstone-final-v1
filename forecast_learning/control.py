"""Use the new predictor only where a feasible route can improve both continuations."""
from functools import lru_cache

import numpy as np

from forecast_avoidance.controller import choose as v18_choose, fragile_margin, onset_mask, proposal_set

from . import ARTIFACTS
from .features import extract
from .protocol import V20_THRESHOLD, read
from .training import ITERATIONS, make_score, restore_model, verify_freeze


GUARDS = dict(target_margin_db=1.5,worst_case_gain_db_seconds=1.5,allowed_early_drop_db=.25,
              per_path_deficit_tolerance_db_seconds=.25,short_score_tolerance=.08,short_score_floor=.18,
              one_relay_only=True,retain_v18_interventions=True)


@lru_cache(maxsize=1)
def load_profile():
    freeze = read(ARTIFACTS/'freeze.json')
    verify_freeze(freeze)
    mode = freeze['control_forecast']
    chosen = freeze['chosen']
    early = restore_model(freeze['models'][chosen['model']]) if mode=='learned' else None
    short = restore_model(freeze['models']['dual_short'])
    return mode, chosen, early, short


def short_score(model, x):
    return float(model.predict(x.reshape(1,-1),num_iteration=ITERATIONS[-1],num_threads=1)[0])


def window_deficit(margins):
    return np.maximum(0.,GUARDS['target_margin_db']-margins[:,6:15]).sum(axis=1)*.5


def choose(profile, planner, actions, logits, valid, local, phase, now):
    original, detail = v18_choose('joint_geometry',planner,actions,logits,valid)
    if not np.array_equal(original,actions):
        return original,detail,np.asarray(detail['projected_fleet_endpoint']),'retained_v18'
    if not planner.state.local_rf_available or phase==2 or not (valid.sum(axis=1)>1).any():
        return actions,None,None,'missing_rf_or_no_alternative'
    x, warning, margins, _, endpoint = extract(planner,actions,local,phase)
    mode, config, early, short = profile
    score = float(make_score(early,config,x.reshape(1,-1),warning.shield_score)[0]) if mode=='learned' else warning.shield_score
    threshold = config['threshold'] if mode=='learned' else V20_THRESHOLD
    if score<threshold:
        return actions,None,None,'below_advance_threshold'
    baseline_deficit = window_deficit(margins)
    gain_min = GUARDS['worst_case_gain_db_seconds']
    if margins[0,0]<0 or baseline_deficit.max()<gain_min:
        return actions,None,None,'no_physical_prevention_evidence'
    base_short = short_score(short,x)
    counts = onset_mask(margins).sum(axis=-1)
    ns = planner.design.n_search
    best = None
    for candidate, relay in proposal_set(actions,logits,valid):
        if relay is None:
            continue
        alt_x,_,alt,graph,alt_endpoint = extract(planner,candidate,local,phase)
        if np.array_equal(alt_endpoint,endpoint):
            continue
        if (alt[:,1:7]<np.minimum(0.,margins[:,1:7])-GUARDS['allowed_early_drop_db']).any():
            continue
        if (onset_mask(alt).sum(axis=-1)>counts).any():
            continue
        cost = window_deficit(alt)
        gain = float(baseline_deficit.max()-cost.max())
        if gain<gain_min or (cost>baseline_deficit+GUARDS['per_path_deficit_tolerance_db_seconds']).any():
            continue
        new_short = short_score(short,alt_x)
        if new_short>max(GUARDS['short_score_floor'],base_short+GUARDS['short_score_tolerance']):
            continue
        worst = int(np.argmax(baseline_deficit))
        window = np.argsort(margins[worst,6:15])[:3]+6
        backup = float(fragile_margin(graph[worst,window],ns).min())
        command_distance = np.linalg.norm(alt_endpoint-endpoint,axis=-1).sum()
        value = gain+.10*np.clip(backup,-15.,15.)-.02*command_distance
        if best is None or value>best[0]:
            best = value,candidate,alt_endpoint,relay,gain,backup,new_short
    if best is None:
        return actions,None,None,'no_robust_safe_gain'
    _,candidate,alt_endpoint,relay,gain,backup,new_short = best
    event = dict(reason='prevention',expert='v21_'+mode,changed_relay=ns+relay,base_action=int(actions[relay]),
                 action=int(candidate[relay]),early_score=score,threshold=threshold,
                 v20_score=warning.shield_score,
                 projected_onset_time_s=now+warning.onset_in_s if np.isfinite(warning.onset_in_s) else None,
                 modeled_short_score_before=base_short,modeled_short_score_after=new_short,
                 robust_gain_db_seconds=gain,backup_margin_db=backup,projected_fleet_endpoint=alt_endpoint.tolist())
    return candidate,event,alt_endpoint,'added_prevention'
