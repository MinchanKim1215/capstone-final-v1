"""Group consecutive one-second forecasts by the predicted absolute onset."""
from dataclasses import dataclass

import numpy as np

from . import ARTIFACTS
from .evaluate import load, paired, summarize
from .protocol import CAP, now, read, sha, write


@dataclass
class LeadTracker:
    threshold: float
    previous_time: float = -np.inf
    previous_target: float = -np.inf
    last_alert_time: float = -np.inf
    last_alert_target: float = -np.inf

    def update(self, time_s, score, onset_in_s):
        """Issue once two consecutive predictions agree about a future outage."""
        if time_s <= self.previous_time:
            raise ValueError('Forecast timestamps must increase')
        target = time_s+onset_in_s if score>=self.threshold and 3<=onset_in_s<7 else -np.inf
        agrees = (np.isfinite(target) and time_s-self.previous_time==1.
                  and abs(target-self.previous_target)<=1.)
        self.previous_time, self.previous_target = time_s, target
        if not agrees or time_s-self.last_alert_time<3. or abs(target-self.last_alert_target)<=2.:
            return False
        self.last_alert_time, self.last_alert_target = time_s, target
        return True


def track(cohort, threshold):
    output = []
    for row,scores in cohort:
        tracker = LeadTracker(threshold)
        clock, raw, lead = [scores[key] for key in ('time_s','shield','onset_in_s')]
        values = [tracker.update(t,r,h) for t,r,h in zip(clock,raw,lead)]
        output.append((row,dict(scores,shield=np.where(np.isfinite(raw),values,np.nan))))
    return output


def lock_profile():
    path = ARTIFACTS/'tracking_freeze.json'
    if path.exists() or (ARTIFACTS/'final_results.json').exists():
        raise ValueError('A new tracker must be locked before reviewing fresh results')
    tracker = LeadTracker(.6)
    assert not tracker.update(21,.9,6.)
    assert tracker.update(22,.9,5.)
    assert not tracker.update(23,.9,4.)
    assert not tracker.update(24,.9,3.)
    assert not tracker.update(25,.9,6.)
    assert tracker.update(26,.9,5.)
    manifests = {name:ARTIFACTS/(name+'_manifest.json') for name in ('calibration','selection')}
    cohorts = {key:[load(row) for row in read(meta)['rows']] for key,meta in manifests.items()}
    scores = []
    for value in [*np.linspace(.05,1.,20),1.01]:
        cal, dev = [summarize(track(cohorts[k],value),'shield_1s',.5) for k in ('calibration','selection')]
        scores.append((float(value),cal,dev))
    eligible = [(q,cal,dev) for q,cal,dev in scores if max(cal['false_per_hour'],dev['false_per_hour'])<=CAP]
    threshold, cal, dev = max(eligible,key=lambda row:(row[1]['recall'],row[1]['precision'],
                              -row[1]['false_per_hour'],row[0]))
    write(path,dict(threshold=threshold,calibration=cal,selection=dev,cost_cap=CAP,
                    selection_rule='Calibration recall among operating points within the cap on both development cohorts.',
                    manifests={key:sha(meta) for key,meta in manifests.items()},
                    source_sha256=sha(__file__),same_forecasts_as_primary=True,created_at=now()))


def final_evaluation():
    path = ARTIFACTS/'tracking_freeze.json'
    frozen = read(path)
    final = read(ARTIFACTS/'final_results.json')
    if sha(__file__) != frozen['source_sha256'] or frozen['created_at'] >= final['created_at']:
        raise ValueError('Tracker must be unchanged from before the fresh comparison')
    for name,expected in frozen['manifests'].items():
        if sha(ARTIFACTS/(name+'_manifest.json'))!=expected:
            raise ValueError('Tracker development evidence changed')
    manifest = ARTIFACTS/'fresh_manifest.json'
    cohort = [load(row) for row in read(manifest)['rows']]
    tracked = track(cohort,frozen['threshold'])
    result = dict(**summarize(tracked,'shield_1s',.5,True),threshold=frozen['threshold'],
                  paired=paired(tracked,'shield_1s',.5), freeze_sha256=sha(path),fresh_manifest_sha256=sha(manifest))
    write(ARTIFACTS/'tracking_result.json',result)
    return result


if __name__=='__main__':
    lock_profile()
