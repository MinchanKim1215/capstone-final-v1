"""Run the Final V1 submission with eight aircraft and a map-based relay plan."""

from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

from fast_halfsec import LEGACY, context
from swarm_sim.mission import MAX_MISSION

from . import DECISION_S, planned_eight
from .planned_eight_motion import HeightNavigator, HeightWindow, VERTICAL_ACCEL_MPS2, VERTICAL_SPEED_MPS
from .release import DEFAULT_SPEC, EXPANDED_REGISTRY, load_model


MODE = 'halfsec_planned_eight'


def map_registry(region):
    """Prefer a local expanded scene, then use the installed team's registry."""
    if EXPANDED_REGISTRY.is_file():
        entry = planned_eight.registry().get(region)
        if entry is not None and (LEGACY / entry['path']).is_file():
            return nullcontext()
    if region not in context.registry():
        raise ValueError(f'Map is not registered in this installation: {region}')
    return patch.object(planned_eight, 'registry', context.registry)


class SubmissionNavigator(HeightNavigator):
    def approach(self, now):
        # A new window can begin while the preceding descent is still in progress.
        # Resolve that proof before requesting another climb so the relay can rejoin.
        if self.prove is not None:
            self.return_to_base(now)
        super().approach(now)

    def altitude_request(self, height, kind):
        if self.prove is None:
            super().altitude_request(height, kind)


class SubmissionFlight(planned_eight.PlannedFlight):
    def navigator(self, city, ns, survey):
        windows = [HeightWindow(**window) for window in self.plan['windows']]
        station = self.plan['station']
        if station and any(window.relay == station['relay'] for window in windows):
            raise ValueError('The station and height roles require different aircraft')
        self.nav = SubmissionNavigator(city, ns, survey, windows, self.confirm, self.descent_safe)
        return self.nav


def run(spec, max_seconds=MAX_MISSION, command_loss=False):
    torch.set_num_threads(1)
    with map_registry(spec['region']):
        plan = planned_eight.plan_for(spec['region'], spec['direction'])
        flight = SubmissionFlight(*load_model(), plan=plan)
        row, frames = flight.run(spec, max_seconds, command_loss)
    nav = flight.nav
    altitudes = np.asarray(nav.samples[:len(frames['network'])], dtype=np.float32)
    if row['total_drones'] != 8 or row['search_drones'] != 3 or altitudes.shape != (len(frames['network']), 8):
        raise ValueError('The flight must preserve the eight-aircraft deployment and altitude trace')
    station = plan['station']
    if station is not None and not np.allclose(frames['positions'][:, station['relay']], station['xy_m'], atol=1e-5, rtol=0.):
        raise ValueError('The fixed relay left its preflight position')
    if nav.max_speed > VERTICAL_SPEED_MPS+1e-6 or nav.max_acceleration > VERTICAL_ACCEL_MPS2+1e-6:
        raise ValueError('The simulated vertical limits were exceeded')
    frames['altitudes'] = altitudes
    row.update(controller=MODE, placement=plan, altitude_windows=nav.windows_log, altitude_moves=nav.moves,
               end_altitudes_m=nav.heights.tolist(), max_vertical_speed_mps=nav.max_speed,
               max_vertical_accel_mps2=nav.max_acceleration)
    return row, frames


def cli_result(spec, steps, command_loss):
    duration = MAX_MISSION if steps is None else steps * DECISION_S
    row, _ = run(spec, duration, command_loss)
    plan = row['placement']
    return dict(
        version='Final V1', controller=MODE, arm=MODE, spec=spec,
        transport='routed_with_forced_loss' if command_loss else 'routed', deployment=row['deployment'],
        total_drones=row['total_drones'], search_drones=row['search_drones'],
        decision_interval_s=row['decision_interval_s'], duration_s=row['duration_s'],
        mission_completed=row['mission_completed'], coverage=row['coverage'],
        waypoints_reached=row['waypoints_reached'], waypoints_total=row['waypoints_total'],
        persistent_onsets=row['onsets'], persistent_cut_seconds=row['cut_seconds'], timely_pdr=row['timely_pdr'],
        transformer_evaluations=row['transformer_evaluations'], cleared_stages=row['cleared_stages'],
        recovery_filtered=row['recovery_filtered'],
        preflight=dict(station=plan['station'], altitude_windows=plan['windows'], reason=plan.get('reason'),
                       nominal_before=plan.get('before'), nominal_after=plan.get('after')),
        verified_crossings=sum(window['outcome'] == 'crossed' for window in row['altitude_windows']),
        end_altitudes_m=row['end_altitudes_m'],
        audited=dict(passed=row['safety_passed'] and row['packet_conserved'],
                     minimum_swept_separation_m=row['min_separation_m'], maximum_speed_mps=row['max_speed_mps'],
                     maximum_vertical_speed_mps=row['max_vertical_speed_mps'], packet_conserved=row['packet_conserved']),
    )
