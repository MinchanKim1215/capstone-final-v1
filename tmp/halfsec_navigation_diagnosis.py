"""Inspect train-side navigation stalls without using confirmation outcomes."""
from pathlib import Path

import numpy as np

from fast_halfsec import ARTIFACTS
from fast_halfsec.context import survey_for
from final_v1_fast.motion import duration
from forecast_rebuild.contracts import read, write


def main():
    source = ARTIFACTS / 'train_manifest.json'
    records = [read(item['metadata']) for item in read(source)['records']]
    failures = []
    for row in records:
        if row['mission_completed']:
            continue
        spec = row['spec']
        city, _, fleet, survey, _ = survey_for(spec['region'], spec['direction'])
        stage = row['waypoints_reached']
        with np.load(row['archive_path'], allow_pickle=False) as data:
            tail = data['positions'][-120:]
        motion_range = np.linalg.norm(tail.max(axis=0)-tail.min(axis=0), axis=-1)
        targets = city.points[survey.waypoints[min(stage,len(survey.waypoints)-1), :fleet.ns]]
        waiting = np.linalg.norm(tail[-1,:fleet.ns]-targets,axis=-1) >= .1
        stuck = motion_range[fleet.ns:] < .1
        dist = np.linalg.norm(targets[:,None]-tail[-1,None,fleet.ns:],axis=-1)
        blockers = np.argwhere(waiting[:,None] & stuck[None,:] & (dist < 5.26))
        search_nodes = np.vstack((survey.initial[:fleet.ns],survey.waypoints[:,:fleet.ns]))
        lengths = np.linalg.norm(np.diff(city.points[search_nodes],axis=0),axis=-1)
        lower_bound = 30. + sum(duration(float(length)) for length in lengths.max(axis=1))
        findings = dict(region=spec['region'], mode=row['mode'], coverage=row['coverage'],
                        nominal_s=row['nominal_duration_s'], simulated_s=row['duration_s'],
                        rest_to_rest_schedule_lower_bound_s=lower_bound,
                        waypoint=stage,total_waypoints=len(survey.waypoints),
                        last_minute_search_range_m=motion_range[:fleet.ns].tolist(),
                        stationary_relay_on_waiting_search_goal=blockers.tolist())
        failures.append(findings)
        print(findings,flush=True)
    write(ARTIFACTS/'navigation_diagnosis_train.json',dict(train_flights=len(records), incomplete=len(failures),
          notes='Descriptive evidence from training geography only. The rest-to-rest lower bound excludes traffic and waits.',
          failures=failures))


if __name__=='__main__':
    main()
