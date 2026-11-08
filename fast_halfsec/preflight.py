"""Measure arrival-paced survey completion on training geography."""
from time import perf_counter

import numpy as np

from forecast_rebuild.contracts import write
from . import ARTIFACTS
from .context import registry, survey_for
from .navigation import Navigator


def navigation_check():
    choices = [entry for entry in registry().values() if entry['study_role'] == 'train' and entry['new_map']]
    checks = []
    for province in ('seoul', 'daegu', 'incheon'):
        region = next(entry['key'] for entry in choices if entry['province'] == province)
        began = perf_counter()
        city, _, fleet, survey, _ = survey_for(region, 'east_west')
        nav = Navigator(city, fleet.ns, survey)
        offsets = np.zeros(fleet.nr, dtype=np.int32)
        for tick in range(72000):
            nav.update(tick * .1)
            if nav.finished_at is not None:
                break
            nav.step(offsets)
        row = dict(region=region, deployment=fleet.key, arrived=nav.stage, waypoints=len(survey.waypoints),
                   nominal_s=survey.baseline_s, completed_s=nav.finished_at,
                   covered=int(nav.covered.sum()), observable=survey.observable, wall_s=perf_counter()-began)
        checks.append(row)
        print(row, flush=True)
    write(ARTIFACTS / 'navigation_preflight.json', checks)
    if any(row['completed_s'] is None or row['covered'] != row['observable'] for row in checks):
        raise ValueError('A new-map survey did not complete in the motion preflight')


if __name__ == '__main__':
    navigation_check()
