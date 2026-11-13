"""Replay one completed test condition to separate forecast and PPO effects."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_halfsec.navigation import Navigator
from final_v1.halfsec_joint import ARTIFACTS
from final_v1.halfsec_joint.mission import JointFlight
from final_v1.halfsec_joint.model import JointPolicy
from final_v1.halfsec_joint.run import load_state
from forecast_rebuild.contracts import now, read, sha, write
from swarm_sim.mission import MAX_MISSION


SCENE = 'busan_20261031_04'
PROBE_S = 505.5


class Stages(Navigator):
    def __init__(self, *args):
        super().__init__(*args)
        self.stages = []

    def update(self, time_s):
        super().update(time_s)
        if abs(time_s * 2. - round(time_s * 2.)) < 1e-8:
            self.stages.append(self.stage)


class Probe(JointFlight):
    def __init__(self, model, threshold, tag):
        super().__init__(model, threshold, False, None)
        self.navigator_factory = Stages
        self.tag = tag
        self.at_probe = None

    def options_for(self, *args):
        options, count = super().options_for(*args)
        self.probe_options = options
        return options, count

    def select(self, policy, inputs, explore, rng):
        selected, value, logp = super().select(policy, inputs, explore, rng)
        if self.frames[-1] == round(PROBE_S * 2.):
            with torch.no_grad():
                logits, _ = self.model.policy(*(torch.from_numpy(a[None]) for a in inputs), self.context)
            self.at_probe = dict(time_s=PROBE_S, stage=self.nav.stage, score=self.scores[-1],
                                 threshold=self.threshold, selected=selected,
                                 choices=[dict(event=option.event, prior=float(inputs[2][i]),
                                               logit=float(logits[0, i]))
                                          for i, option in enumerate(self.probe_options)])
            chosen = self.probe_options[selected]
            print(self.tag, 'at', PROBE_S, 'risk', round(self.scores[-1], 5),
                  'chosen', None if chosen.event is None else chosen.event['new_offset'], flush=True)
        return selected, value, logp


def replay(job):
    tag, spec, forecast, policy, reference_stage = job
    torch.set_num_threads(1)
    neural, actor = load_state(forecast), load_state(policy)
    if neural['threshold'] != actor['threshold']:
        raise ValueError('The two checkpoints have different operational thresholds')
    model = JointPolicy().eval()
    model.load_state_dict(neural['model'])
    model.moves.load_state_dict({k.removeprefix('moves.'): v for k, v in actor['model'].items()
                                 if k.startswith('moves.')})
    probe = Probe(model, neural['threshold'], tag)
    row, arrays = probe.run(spec, .5, MAX_MISSION)
    stage = np.asarray(probe.nav.stages[:len(arrays['network'])])
    if len(stage) != len(arrays['network']):
        raise ValueError('The stage and RF clocks did not align')
    beyond = np.flatnonzero(stage > reference_stage)
    cuts = [(start, end, int(stage[round(start * 2)])) for start, end in row['cuts']]
    result = {k:row[k] for k in ('mission_completed','duration_s','coverage','waypoints_reached',
                                'waypoints_total','onsets','cut_seconds','cuts','safety_passed',
                                'packet_conserved','wall_s')}
    result.update(forecast=forecast['step'], ppo=policy['step'], event_probe=probe.at_probe,
                  first_beyond_reference_s=float(beyond[0] / 2.) if len(beyond) else None,
                  onsets_beyond_reference=sum(t > 0. and s > reference_stage for t, _, s in cuts),
                  cut_stages=cuts)
    return tag, result


def main():
    result = read(ARTIFACTS / 'results.json')
    if sha(ARTIFACTS / 'freeze.json') != result['frozen_sha256']:
        raise ValueError('The previous comparison is not intact')
    freeze = read(ARTIFACTS / 'freeze.json')
    old, trained = freeze['initial'], freeze['selected']
    rows = {}
    for path in (ARTIFACTS / 'flights/confirmation').glob('*.json'):
        row = read(path)
        if row['spec']['region'] == SCENE and row['spec']['kind'] == 'continuous':
            rows[row['arm']] = row
    warm, joint = rows['warm_05s'], rows['joint_05s']
    assert warm['spec'] == joint['spec']
    spec = warm['spec']
    stage = warm['waypoints_reached']
    jobs = [(tag, spec, forecast, policy, stage) for tag, forecast, policy in (
        ('warm', old, old), ('forecast_only', trained, old),
        ('ppo_only', old, trained), ('joint', trained, trained))]
    results = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            tag, values = future.result()
            results[tag] = values
            print(tag, {k:values[k] for k in ('mission_completed','onsets','cut_seconds','duration_s',
                                             'waypoints_reached','first_beyond_reference_s',
                                             'onsets_beyond_reference','wall_s')}, flush=True)
    for tag, previous in (('warm', warm), ('joint', joint)):
        for name in ('mission_completed', 'onsets', 'cut_seconds', 'duration_s', 'cuts'):
            if previous[name] != results[tag][name]:
                raise ValueError(f'The {tag} replay differs from the completed evaluation: {name}')
    write(ARTIFACTS / 'busan_diagnostic_20261110.json',
          dict(created_at=now(), scope='post-evaluation paired replay of one prior condition',
               study_sha256=sha(ARTIFACTS / 'results.json'), spec=spec, warm_reference_stage=stage,
               parity_with_archives=True, cases=results))


if __name__ == '__main__':
    main()
