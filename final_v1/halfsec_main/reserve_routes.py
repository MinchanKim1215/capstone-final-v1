"""Trial a make-before-break relay move using forecast paths and received RF only."""

from functools import lru_cache
from unittest.mock import patch

import numpy as np

from adaptive_joint_expanded.features import OFFSETS, widest
from fast_halfsec.controller import ADJACENT, PLANNER_DELAY_TICKS, Option
from fast_halfsec.features import MAX_OPTIONS, N_CANDIDATE
from final_v1_fast.controller import TIMES, margins, runs
from swarm_sim.mission import MAX_MISSION

from .mission import MainFlight
from .release import load_model, map_scope


FUTURE = np.diff(TIMES, append=TIMES[-1])
MIN_RESERVE_GAIN_S = .5


@lru_cache(maxsize=None)
def failure_masks(nodes, searchers):
    """Remove each relay or one direct search-to-search link at a time."""
    source, target = np.triu_indices(searchers, k=1)
    masks = np.zeros((nodes - searchers + len(source), nodes, nodes), bool)
    for i, relay in enumerate(range(searchers, nodes)):
        masks[i, relay, :] = masks[i, :, relay] = True
    index = np.arange(nodes - searchers, len(masks))
    masks[index, source, target] = masks[index, target, source] = True
    return masks


def reserve_margin(graphs, searchers):
    masks = failure_masks(graphs.shape[-1], searchers)
    remaining = np.where(masks[:, None], -np.inf, graphs[None])
    return widest(remaining, searchers).min(axis=(0, 2))


def rank_move(base_pairs, base_reserve, pairs, reserve, displacement):
    """Improve forecast redundancy without losing an existing search connection."""
    if np.any((pairs < 0.) & (base_pairs >= 0.)):
        return None
    gained = float(np.dot((reserve >= 0.) * 1. - (base_reserve >= 0.), FUTURE))
    if gained < MIN_RESERVE_GAIN_S - 1e-8:
        return None
    connected_gain = float(np.dot((pairs.min(axis=1) >= 0.) * 1. -
                                  (base_pairs.min(axis=1) >= 0.), FUTURE))
    reserve_gain = float(np.dot(np.clip(1. - base_reserve, 0., 10.) -
                                np.clip(1. - reserve, 0., 10.), FUTURE))
    return connected_gain, gained, reserve_gain, -displacement


class ReserveRouteFlight(MainFlight):
    def __init__(self, model, threshold):
        super().__init__(model, threshold)
        self.prefer_reserve = False
        self.reserve_trigger_ticks = 0

    def reserve_options(self, nav, scenario, now, requested, applied, estimate, state, delivery):
        if nav.clearing or nav.finished_at is not None or state[1][0] < 0.:
            return [], 0
        _, base, graphs, hold_path = state
        if not np.array_equal(requested, applied):
            hold_path = nav.forecast(requested, applied, delivery)
            base, graphs = margins(nav.city, scenario, hold_path, estimate, nav.ns)
        reference = reserve_margin(graphs, nav.ns)
        fragile = runs(reference)
        if not fragile:
            return [], 0
        self.reserve_trigger_ticks += 1
        base_pairs = widest(graphs, nav.ns)
        original_cuts = runs(base)
        proposals = []
        seen = {hold_path.tobytes()}
        for relay, offset in enumerate(requested):
            for target in ADJACENT[offset]:
                change = requested.copy()
                change[relay] = target
                path = nav.forecast(change, applied, PLANNER_DELAY_TICKS + delivery)
                signature = path.tobytes()
                if signature in seen:
                    continue
                seen.add(signature)
                _, alternative = margins(nav.city, scenario, path, estimate, nav.ns)
                pairs = widest(alternative, nav.ns)
                reserve = reserve_margin(alternative, nav.ns)
                displacement = float(np.linalg.norm(path[-1] - hold_path[-1], axis=-1).sum())
                rank = rank_move(base_pairs, reference, pairs, reserve, displacement)
                if rank is None:
                    continue
                event = dict(time_s=now, changed_relay=nav.ns + relay, reason='reserve_route',
                             old_offset=OFFSETS[offset].tolist(), new_offset=OFFSETS[target].tolist(),
                             modeled_onsets_s=original_cuts, reserve_loss_in_s=fragile[0],
                             modeled_connection_gain_s=rank[0], modeled_reserve_gain_s=rank[1])
                proposals.append((rank, Option(change, np.zeros(N_CANDIDATE, np.float32), rank[2], event)))
        if not proposals:
            return [], len(seen) - 1
        proposals.sort(key=lambda item: item[0], reverse=True)
        values = np.zeros(N_CANDIDATE, np.float32)
        values[-1] = 1.
        hold = Option(requested.copy(), values, 0., None)
        return [hold, *(item[1] for item in proposals[:MAX_OPTIONS - 1])], len(seen) - 1

    def options_for(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        self.prefer_reserve = False
        proposed, tested = self.reserve_options(nav, scenario, now, requested, applied, estimate, state, delivery)
        args = (nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery)
        if proposed:
            # Use the release's same navigation, impending-risk and recovery guards.
            with patch.object(self, 'physical_options', return_value=(proposed, tested)):
                filtered, _ = super().options_for(*args)
            if len(filtered) > 1:
                self.prefer_reserve = True
                return filtered, tested
        options, standard_tested = super().options_for(*args)
        return options, tested + standard_tested

    def select(self, policy, inputs, explore, rng):
        if self.prefer_reserve:
            return 1, 0., 0.
        return super().select(policy, inputs, explore, rng)


def run(spec, max_seconds=MAX_MISSION):
    model, threshold = load_model()
    with map_scope(spec['region']):
        flight = ReserveRouteFlight(model, threshold)
        row, frames = flight.run(spec, max_seconds)
    row['controller'] = 'halfsec_reserve_routes'
    row['reserve_trigger_ticks'] = flight.reserve_trigger_ticks
    row['reserve_requests'] = sum(event['reason'] == 'reserve_route' for event in row['events'])
    return row, frames
