"""Load the pinned inference model and run the default Final V1 simulation."""

from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch

import torch

from fast_halfsec import context
from final_v1.halfsec_joint.model import JointPolicy
from forecast_rebuild.contracts import read, sha
from swarm_sim.mission import MAX_MISSION

from . import DECISION_S, MODE, ROOT
from .mission import MainFlight


WEIGHTS = ROOT / 'checkpoint.pt'
MANIFEST = ROOT / 'checkpoint.json'
EXPANDED_REGISTRY = ROOT.parent / 'halfsec_expanded/artifacts/map_registry.json'
# An established confirmation case supported by the team download's map registry.
DEFAULT_SPEC = dict(region='gwangju_20261104_b2_ba3bb4f31d', direction='north_south',
                    seed=1591375443, kind='clear')


@lru_cache(maxsize=1)
def expanded_plan():
    return read(EXPANDED_REGISTRY)


@contextmanager
def map_scope(region):
    if region in context.registry() or not EXPANDED_REGISTRY.is_file():
        yield
        return
    original_plan = context.contracts.plan
    original_model, original_config = context.legacy.MODEL, context.legacy.CONFIG
    original_paths = [(module, module.ARTIFACTS)
                      for module in (context.baseline, context.map_features, context.world)]
    context.registry.cache_clear()
    context.survey_for.cache_clear()
    try:
        with patch.multiple(context, study_plan=expanded_plan, REGISTRY=EXPANDED_REGISTRY, READY=False):
            yield
    finally:
        context.contracts.plan = original_plan
        context.legacy.MODEL, context.legacy.CONFIG = original_model, original_config
        for module, path in original_paths:
            module.ARTIFACTS = path
        context.registry.cache_clear()
        context.survey_for.cache_clear()


@lru_cache(maxsize=1)
def load_model():
    record = read(MANIFEST)
    if record['controller'] != MODE or record['decision_interval_s'] != DECISION_S or sha(WEIGHTS) != record['sha256']:
        raise ValueError('The Final V1 half-second release checkpoint has changed')
    values = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    if values['step'] != record['step'] or values['threshold'] != record['threshold']:
        raise ValueError('The weights and operating point disagree with the release manifest')
    model = JointPolicy()
    model.load_state_dict(values['model'], strict=True)
    return model.eval(), float(values['threshold'])


def run(spec, max_seconds=MAX_MISSION, command_loss=False):
    model, threshold = load_model()
    with map_scope(spec['region']):
        row, frames = MainFlight(model, threshold).run(spec, max_seconds, command_loss)
    row['controller'] = MODE
    return row, frames


def cli_result(spec, steps, command_loss):
    duration = MAX_MISSION if steps is None else steps * DECISION_S
    row, _ = run(spec, duration, command_loss)
    return dict(
        version='Final V1', controller=MODE, arm=MODE, spec=spec,
        transport='routed_with_forced_loss' if command_loss else 'routed',
        deployment=row['deployment'], decision_interval_s=row['decision_interval_s'],
        duration_s=row['duration_s'], mission_completed=row['mission_completed'],
        waypoints_reached=row['waypoints_reached'], waypoints_total=row['waypoints_total'],
        persistent_onsets=row['onsets'], persistent_cut_seconds=row['cut_seconds'], timely_pdr=row['timely_pdr'],
        transformer_evaluations=row['transformer_evaluations'], cleared_stages=row['cleared_stages'],
        recovery_filtered=row['recovery_filtered'],
        audited=dict(passed=row['safety_passed'] and row['packet_conserved'],
                     minimum_swept_separation_m=row['min_separation_m'],
                     maximum_speed_mps=row['max_speed_mps'], packet_conserved=row['packet_conserved']),
    )
