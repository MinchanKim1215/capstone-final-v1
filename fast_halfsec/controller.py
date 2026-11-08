"""Evaluate safe single-relay changes at the current planning tick."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from adaptive_joint_expanded.features import OFFSETS, widest
from final_v1_fast.controller import TIMES, critical_endpoints, margins, runs
from .features import EARLY, MAX_OPTIONS, N_CANDIDATE, deficit, extract, physical_risk

EARLY_SAFE = TIMES <= 3.
MIN_GAIN = 1.
EARLY_TOLERANCE = .25
PLANNER_DELAY_TICKS = 2
COMMAND_DELAY_TICKS = 1
ADJACENT = tuple(tuple(sorted({0, *np.flatnonzero(np.abs(OFFSETS-offset).sum(axis=1) == 20.)} - {i}))
                 for i, offset in enumerate(OFFSETS))


@dataclass
class Option:
    offsets: np.ndarray
    values: np.ndarray
    physical_value: float
    event: dict | None


def state_at(nav, scenario, now, applied, estimate, history):
    xy = nav.forecast(applied, applied, commit_frames=0)
    margin, graph = margins(nav.city, scenario, xy, estimate, nav.ns)
    raw = extract(margin, estimate, nav.ns, history, now, nav.stage/len(nav.survey.waypoints),
                  nav.finished_at is not None)
    return raw, margin, graph, xy


def backup_margin(graph, ns):
    a, b = critical_endpoints(graph, ns)
    other = graph.copy()
    other[a, b] = other[b, a] = -100.
    return float(widest(other, ns).min())


def options_for(nav, scenario, now, requested, applied, estimate, state, learned_risk=False,
                delivery_ticks=COMMAND_DELAY_TICKS):
    _, base, graph, base_xy = state
    base_runs = runs(base)
    loss = deficit(base)
    keep = np.zeros(N_CANDIDATE, np.float32)
    keep[-1] = 1.
    hold = Option(requested.copy(), keep, 0., None)
    if nav.finished_at is not None or not estimate.local_rf:
        return [hold], 0
    if (not base_runs or loss < MIN_GAIN) and not (learned_risk and base[EARLY].min() < 6.):
        return [hold], 0
    candidates = []
    unique = {base_xy.tobytes()}
    tested = 0
    worst = int(np.argmin(base))
    backup = backup_margin(graph[worst], nav.ns)
    # Candidate forecasts include the same published-motion and delivery delay
    # used by the simulated command channel.
    for relay, current in enumerate(requested):
        for action in ADJACENT[current]:
            target = requested.copy()
            target[relay] = action
            xy = nav.forecast(target, applied, PLANNER_DELAY_TICKS + delivery_ticks)
            signature = xy.tobytes()
            if signature in unique:
                continue
            unique.add(signature)
            tested += 1
            alt, links = margins(nav.city, scenario, xy, estimate, nav.ns)
            if (alt[EARLY_SAFE] < np.minimum(0., base[EARLY_SAFE]) - EARLY_TOLERANCE).any():
                continue
            alt_runs = runs(alt)
            if len(alt_runs) > len(base_runs):
                continue
            gain = loss - deficit(alt)
            if gain < MIN_GAIN:
                continue
            alt_backup = backup_margin(links[worst], nav.ns)
            change = float(np.linalg.norm(OFFSETS[action] - OFFSETS[current]))
            distance = float(np.linalg.norm(xy[-1] - base_xy[-1], axis=-1).sum())
            value = gain - .02 * distance - .10 * change / 20.
            fields = np.asarray([gain/20., base.min()/20., alt.min()/20.,
                                 (alt[EARLY_SAFE].min()-base[EARLY_SAFE].min())/20.,
                                 len(base_runs)-len(alt_runs), value/20., change/60., distance/80.,
                                 relay/max(1, len(requested)-1), (alt_backup-backup)/20., 0.], np.float32)
            event = dict(time_s=now, changed_relay=nav.ns+relay,
                         old_offset=OFFSETS[current].tolist(), new_offset=OFFSETS[action].tolist(),
                         modeled_onsets_s=base_runs, modeled_gain_db_s=gain,
                         worst_backup_delta_db=alt_backup-backup, physics_value=value,
                         reason='prevention' if base[0] >= 0 else 'recovery')
            candidates.append(Option(target, np.clip(fields, -8., 8.), value, event))
    candidates.sort(key=lambda candidate: candidate.physical_value, reverse=True)
    return [hold, *candidates[:MAX_OPTIONS-1]], tested
