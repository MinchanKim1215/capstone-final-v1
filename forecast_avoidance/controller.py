"""Resolve proposed fleet motion before scoring the resulting search network."""
import numpy as np

from adaptive_joint_expanded.features import OFFSETS, THRESHOLD, widest
from forecast_navigation.preview import preview
from forecast_prevention.controller import spatial_loss


TIMES = np.arange(18) * .5
LEFT = np.arange(18) // 6
FRACTION = (np.arange(18) % 6)[:, None, None] / 6


def samples(knots):
    return knots[LEFT] * (1 - FRACTION) + knots[LEFT + 1] * FRACTION


def onset_mask(margins):
    down = margins < 0
    # Two consecutive failed half-seconds. Suppress the left-censored run.
    return down[..., 1:-1] & down[..., 2:] & ~down[..., :-2]


def fragile_margin(graph, n_search):
    """Minimum terminal bottleneck after deletion of any single link."""
    i, j = np.triu_indices(graph.shape[-1], 1)
    alternatives = np.repeat(graph[:, None], len(i), axis=1)
    edges = np.arange(len(i))
    alternatives[:, edges, i, j] = -np.inf
    alternatives[:, edges, j, i] = -np.inf
    return widest(alternatives, n_search).min(axis=(-1, -2))


class Planner:
    def __init__(self, city, design, scenario, nav, previous_step, baseline_duration, state):
        self.city = city
        self.design = design
        self.scenario = scenario
        self.nav = nav
        self.previous_step = previous_step
        self.baseline_duration = baseline_duration
        self.state = state

    def path(self, actions):
        correction = np.zeros((self.design.n_drones, 2))
        correction[self.design.n_search:] = OFFSETS[actions]
        return preview(self.city, self.design, self.scenario, dict(self.nav, correction=correction),
                       self.baseline_duration, self.previous_step)

    def radio(self, paths, trend=False):
        xy = np.stack([samples(path) for path in paths])
        loss = spatial_loss(self.city, self.scenario, xy[..., :, None, :], xy[..., None, :, :])
        residual = np.nan_to_num(np.clip(self.state.power - (10. - loss[0, 0]), -5., 5.))
        noise = self.state.noise + np.zeros((len(TIMES), 1))
        if trend:
            noise += np.minimum(5., TIMES[:, None] * self.state.noise_trend.clip(0, 1.25))
        directed = (10. - loss + residual * np.exp(-TIMES[:, None, None] / 4.)
                    - noise[:, None, :] - THRESHOLD)
        graph = np.minimum(directed, directed.swapaxes(-1, -2))
        # First segment is proposed by the actual planner. Following commands are hypotheses.
        graph -= .25 * np.maximum(0., TIMES[:, None, None] - 3.)
        return widest(graph, self.design.n_search).min(axis=-1), graph


def radio_evidence(state, observer):
    elevated = (state.noise >= -103.) & (state.noise_trend >= .4)
    return bool(elevated[observer] and elevated.sum() >= 2)


def proposal_set(base, logits, valid):
    yield base.copy(), None
    for relay in range(len(base)):
        ranked = np.argsort(-logits[relay], kind='stable')
        shortlist = [int(a) for a in ranked if valid[relay, a] and a != base[relay]][:7]
        if base[relay] != 0 and valid[relay, 0] and 0 not in shortlist:
            shortlist.append(0)
        for action in shortlist:
            alternative = base.copy()
            alternative[relay] = action
            yield alternative, relay


def choose(mode, planner, base, logits, valid):
    state = planner.state
    ns = planner.design.n_search
    if not state.local_rf_available or not state.known[:ns].all():
        return base, dict(reason='insufficient_observation')
    original = planner.path(base)
    base_margins, _ = planner.radio([original])
    margin = base_margins[0]
    trend = mode == 'joint_radio' and radio_evidence(state, ns)
    if trend:
        # Keep the geometry model whenever it already forecasts a persistent failure.
        trend = not onset_mask(margin).any() and not (margin[:2] < 0).all()
    if trend:
        base_margins, _ = planner.radio([original], trend=True)
        margin = base_margins[0]
    onsets = TIMES[1:-1][onset_mask(margin)].tolist()
    # The t=0 projection is already too late to count as advance warning.
    recovery = margin[0] < 0
    detail = dict(expert='radio_trend' if trend else 'geometry',
                  predicted_onsets_in_s=onsets, original_margin_db=margin.tolist(),
                  observed_current_margin_db=float(margin[0]), estimated_connected=not recovery)
    if not onsets and not (recovery and (margin[1:6] < 0).sum() >= 2):
        return base, dict(reason='queue_watch' if state.queue_watch else 'no_predicted_cut', **detail)
    paths, actions, relays = [original], [base], [None]
    seen = {original.tobytes()}
    for candidate, relay in proposal_set(base, logits, valid):
        if relay is None:
            continue
        path = planner.path(candidate)
        # Require a realizable change in the issued segment, not a hypothetical later move.
        key = path.tobytes()
        if key in seen or np.array_equal(path[1], original[1]):
            continue
        seen.add(key)
        paths.append(path)
        actions.append(candidate)
        relays.append(relay)
    if len(paths) == 1:
        return base, dict(reason='no_distinct_feasible_move', **detail)
    margins, graphs = planner.radio(paths, trend=trend)
    count = onset_mask(margins).sum(axis=-1)
    bad = (margins[:, 1:] < 0).sum(axis=-1)
    deficit = np.maximum(0., -margins[:, 1:]).sum(axis=-1) * .5
    early_safe = (margins[:, 1:6] >= np.minimum(margins[0, 1:6], 0.) - .25).all(axis=-1)
    if recovery:
        improves = (count <= count[0]) & (bad < bad[0]) & (deficit < deficit[0])
    else:
        improves = (count < count[0]) & (bad < bad[0]) & (deficit < deficit[0])
    options = np.flatnonzero(early_safe & improves)
    if not len(options):
        return base, dict(reason='no_predicted_improvement', **detail)
    # Break comparable primary outcomes in favor of a backup route at the vulnerable time.
    best_count = count[options].min()
    options = options[count[options] == best_count]
    best_bad = bad[options].min()
    options = options[bad[options] == best_bad]
    vulnerable = np.argsort(margins[0, 1:])[:4] + 1
    backup = np.array([fragile_margin(graphs[k, vulnerable], ns).min() for k in options])
    travel = np.array([np.linalg.norm(paths[k][1] - paths[k][0], axis=-1).sum() for k in options])
    cost = deficit[options] + .03 * travel - .1 * np.clip(backup, -15., 15.)
    winner = int(np.argmin(cost))
    best = options[winner]
    relay = relays[best]
    return actions[best], dict(reason='recovery' if recovery else 'prevention', changed_relay=ns+relay,
                               alternative_margin_db=margins[best].tolist(),
                               estimated_cost_reduction=float(deficit[0]-deficit[best]),
                               predicted_backup_margin_db=float(backup[winner]),
                               projected_fleet_endpoint=paths[best][1].tolist(),
                               projected_own_endpoint=paths[best][1, ns+relay].tolist(), **detail)
