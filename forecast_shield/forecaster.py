"""Causal network cut forecasts for a proposed or already issued fleet plan."""
from dataclasses import dataclass

import numpy as np

from adaptive_joint_expanded.features import THRESHOLD, widest
from forecast_avoidance.controller import TIMES, onset_mask
from forecast_prevention.diagnostics import reachability


EARLY = (TIMES[1:-1] >= 3.) & (TIMES[1:-1] < 7.)
STRESSES = np.array([-1.5, -.9, -.45, 0., .45, .9, 1.5])
WEIGHTS = np.array([.06, .12, .19, .26, .19, .12, .06])


@dataclass(frozen=True)
class Warning:
    status: str
    legacy_score: float = np.nan
    planner_score: float = np.nan
    shield_score: float = np.nan
    short_score: float = np.nan
    onset_in_s: float = np.nan
    building_support: float = np.nan
    changing_building_support: float = np.nan
    critical_links: tuple = ()
    missing_search_beacons: int = 0
    unknown_rf_receivers: int = 0


def sample_plan(knots, elapsed):
    elapsed = float(elapsed)
    frames = np.rint((elapsed + TIMES)*2).astype(int)
    if elapsed not in (0., 1., 2.) or knots.ndim != 3 or knots.shape[-1] != 2:
        raise ValueError('Expected a fleet plan at a 1 Hz observation boundary')
    left, index = np.divmod(frames, 6)
    if left[-1]+1 >= len(knots):
        raise ValueError('Published plan does not cover the forecast horizon')
    fraction = index[:, None, None]/6
    return knots[left]*(1.-fraction) + knots[left+1]*fraction


def hop_margin(directed):
    return np.minimum(directed, directed.swapaxes(-1,-2))


def network_margin(graph, ns):
    return widest(graph, ns).min(axis=-1)


def two_healthy(margin):
    up = margin >= 0.
    return up[...,1:-1] & up[...,2:]


def geometry(city, scenario, xy):
    tx, rx = np.broadcast_arrays(xy[:, :, None], xy[:, None, :])
    tx = np.c_[tx.reshape(-1,2), np.full(tx.size//2, scenario.altitude_m)]
    rx = np.c_[rx.reshape(-1,2), np.full(rx.size//2, scenario.altitude_m)]
    loss, _, extra = city.radio.propagation(tx, rx, scenario)
    shape = (len(xy), xy.shape[1], xy.shape[1])
    return loss.reshape(shape), extra.reshape(shape)


def evaluate_terms(loss, extra, state, n_search, elapsed, explain=False):
    """Risk support, not a calibrated probability. The same terms apply to every arm."""
    missing = int((~state.known[:n_search]).sum())
    measured = np.isfinite(state.power).any(axis=0)
    unknown = int((~measured).sum())
    if not state.local_rf_available:
        return Warning('no_recent_rf', missing_search_beacons=missing, unknown_rf_receivers=unknown)
    committed = 3.-elapsed
    beyond_commit = np.maximum(0., TIMES-committed)
    residual = np.nan_to_num(np.clip(state.power-(10.-loss[0]), -5., 5.))
    directed = (10.-loss + residual*np.exp(-TIMES[:,None,None]/4.)
                - state.noise[None,None,:] - THRESHOLD - .25*beyond_commit[:,None,None])
    graph = hop_margin(directed)
    nominal = network_margin(graph, n_search)
    forecast = bool(nominal[0] >= 0 and (onset_mask(nominal) & EARLY).any())
    legacy = float(forecast) if not missing else np.nan
    # Missing RF measurements widen the stress band. Planner positions never substitute for RF.
    spread = np.where(measured, 1.25, 3.)[None,:] + .20*beyond_commit[:,None]
    stressed = directed[None] - STRESSES[:,None,None,None]*spread[None,:,None,:]
    margin = network_margin(hop_margin(stressed), n_search)
    onset = onset_mask(margin) & (margin[:,0,None] >= 0.)
    no_building = network_margin(hop_margin(stressed+extra), n_search)
    previous_building = network_margin(hop_margin(stressed+extra-extra[0]), n_search)
    building = onset & two_healthy(no_building)
    changing = building & two_healthy(previous_building)
    # Retain non-building warnings at lower confidence for a calibrated operating point.
    strength = (.40*onset + .35*building + .25*changing)
    early = strength[:, EARLY]
    score = float(WEIGHTS @ early.max(axis=1))
    short = float(WEIGHTS @ strength[:, :5].max(axis=1))
    onset_in_s = np.nan
    critical = ()
    if score > 0:
        slot = int(np.flatnonzero(EARLY)[np.argmax(WEIGHTS @ early)])+1
        onset_in_s = float(TIMES[slot])
        if explain:
            # Evidence for a fragile transition on the nominal path at the reported time.
            current = graph[slot] >= 0.
            prior = graph[slot-1] >= 0.
            edges = np.argwhere(np.triu(prior & ~current & ~reachability(current), 1))
            critical = tuple((int(i), int(j), float(extra[slot,i,j]-extra[slot-1,i,j])) for i,j in edges)
    return Warning('planner_with_stale_beacons' if missing else 'fresh_beacons', legacy,
                   float(forecast), score, short, onset_in_s,
                   float(WEIGHTS @ building[:,EARLY].any(axis=1)),
                   float(WEIGHTS @ changing[:,EARLY].any(axis=1)), critical, missing, unknown)


def predict(city, scenario, state, n_search, knots, elapsed=0., explain=False):
    """Pass current planner state and only RF reports available at the decision time."""
    xy = sample_plan(np.asarray(knots), elapsed)
    if not state.local_rf_available:
        return Warning('no_recent_rf', missing_search_beacons=int((~state.known[:n_search]).sum()))
    loss, extra = geometry(city, scenario, xy)
    return evaluate_terms(loss, extra, state, n_search, elapsed, explain)
