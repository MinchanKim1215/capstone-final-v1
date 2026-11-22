"""Recovery after sustained loss of previously received search beacons."""

from copy import copy
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import numpy as np

from adaptive_joint_expanded.features import OFFSETS, widest
from forecast_avoidance.controller import TIMES, onset_mask, proposal_set
from forecast_rebuild.contracts import read, sha

from . import ARTIFACTS


MODE = 'v21_transformer_recovery'
RUN_DIR = ARTIFACTS / 'relocation'
NATIVE_RELEASE = ARTIFACTS.parent / 'transformer_upgrade/artifacts/integration.json'
CONTROL_PERIOD_S = 3.


def approved():
    path = RUN_DIR / 'results.json'
    if not path.is_file():
        return False
    try:
        result = read(path)
        return (result['candidate'] == MODE and result['accepted']
                and result['policy_sha256'] == sha(__file__)
                and result['baseline_sha256'] == sha(NATIVE_RELEASE)
                and result['protocol_sha256'] == sha(RUN_DIR / 'protocol.json'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def first_stable(margins):
    up = margins >= 0.
    stable = up[:, 1:-1] & up[:, 2:]
    return np.where(stable.any(axis=1), TIMES[1:-1][stable.argmax(axis=1)], np.inf)


def recovery_move(planner, base, logits, valid, missing, now):
    state = planner.state
    ns = planner.design.n_search
    original = planner.path(base)
    age = state.age[:ns][missing]
    # Bound changes in the offset and reduce the bound as observations age.
    oldest = age.max()
    cap = 80. if oldest <= 6. else 57. if oldest <= 12. else 40.
    allowed = valid & (np.linalg.norm(OFFSETS[None] - OFFSETS[base, None], axis=-1) <= cap)
    allowed &= state.known[ns:, None]
    paths, moves = [original], [(base, None)]
    seen = {original.tobytes()}
    for candidate, relay in proposal_set(base, logits, allowed):
        if relay is None:
            continue
        path = planner.path(candidate)
        key = path.tobytes()
        if (key in seen or np.array_equal(path[1, ns + relay], original[1, ns + relay])
                or not np.array_equal(path[1, :ns], original[1, :ns])):
            continue
        seen.add(key)
        paths.append(path)
        moves.append((candidate, relay))
    if len(paths) == 1:
        return base, None

    shadow = copy(planner)
    noise = state.noise.copy()
    lost = np.flatnonzero(missing)
    noise[lost] = np.maximum(noise[lost], noise[state.known].max())
    shadow.state = replace(state, noise=noise)
    _, graphs = shadow.radio(paths)
    penalty = np.zeros((len(TIMES), len(noise)))
    penalty[:, lost] = .25 * age + .20 * TIMES[:, None]
    margins = widest(graphs - penalty[:, :, None] - penalty[:, None, :], ns)
    source, target = np.triu_indices(ns, 1)
    affected = missing[source] | missing[target]
    reconnect = margins[..., affected].min(axis=-1)
    when = first_stable(reconnect)
    deficit = .5 * np.maximum(-reconnect[:, 1:], 0.).sum(axis=1)

    healthy = margins[:, 1:7, ~affected]
    safe = (healthy >= np.minimum(0., healthy[0])).all(axis=(1, 2))
    # A shorter current cut must not come at the expense of a new modeled cut.
    onsets = onset_mask(margins.min(axis=-1)).sum(axis=1)
    faster = when < when[0]
    less_deficit = (when == when[0]) & (deficit <= deficit[0] - 1.5)
    eligible = np.flatnonzero(safe & (onsets <= onsets[0]) & (faster | less_deficit))
    if not len(eligible):
        return base, None
    travel = np.asarray([np.linalg.norm(path[1] - path[0], axis=-1).sum() for path in paths])
    ranking = np.lexsort((deficit[eligible] + .03 * travel[eligible], when[eligible]))
    best = eligible[ranking[0]]
    move, relay = moves[best]
    return move, dict(
        time_s=now, reason='recovery', expert='persistent_beacon_recovery', changed_relay=ns + relay,
        base_action=int(base[relay]), action=int(move[relay]),
        missing_search=lost.tolist(), beacon_age_s=age.tolist(),
        base_reconnect_s=float(when[0]) if np.isfinite(when[0]) else None,
        projected_reconnect_s=float(when[best]) if np.isfinite(when[best]) else None,
        projected_deficit_improvement_db_s=float(deficit[0] - deficit[best]),
        projected_fleet_endpoint=paths[best][1].tolist(),
    )


class Recovery:
    def __init__(self, choose):
        self.choose = choose
        self.previous_missing = None
        self.previous_time = None

    def __call__(self, arm, planner, actions, logits, valid, local, phase, now,
                 base_data=None, allow_recovery=True):
        state = planner.state
        ns = planner.design.n_search
        missing = ~state.known[:ns] & (state.age[:ns] > 3.)
        consecutive = self.previous_time is not None and now - self.previous_time == CONTROL_PERIOD_S
        persistent = consecutive and np.any(missing & self.previous_missing)
        self.previous_missing, self.previous_time = missing, now
        native = self.choose(arm, planner, actions, logits, valid, local, phase, now, base_data, allow_recovery)
        if (native[1] is not None or not allow_recovery or phase == 2
                or not persistent or not state.known[ns]):
            return native
        move, event = recovery_move(planner, actions, logits, valid, missing, now)
        return (move, event, 'persistent_beacon_recovery') if event is not None else native


@lru_cache(maxsize=1)
def transformer():
    from .transformer_upgrade.release import installed

    payload, freeze = installed()
    return payload, freeze['thresholds']['fleet']


def run(spec, transport=None, max_steps=None):
    from . import runner
    from .transformer_upgrade.bridge import run as run_transformer

    payload, profile = transformer()
    with patch.object(runner, 'choose', Recovery(runner.choose)):
        row, arrays = run_transformer(spec, 'v21_transformer', payload, profile['threshold'],
                                      transport=transport, max_steps=max_steps,
                                      network_weight=profile['network_weight'])
    row['upgrade_arm'] = MODE
    return row, arrays
