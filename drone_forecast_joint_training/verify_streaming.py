"""Replay a delivered deterministic validation flight through the public inference API."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
import argparse,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from solo.contracts import ROOT,atomic_json,deterministic,sha
from solo.deploy import StreamingController
from solo.training import best_path
from solo.data import read_episode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='verified_joint');args=parser.parse_args()
    deterministic(True);checkpoint=best_path(ROOT/'results/transformer/train'/args.run)
    summary_path=ROOT/'results/transformer/evaluation/validation'/sha(checkpoint)[:16]/'summary.json'
    summary=json.loads(summary_path.read_text());prefix=ROOT/summary['raw_paths'][0]
    keys=['rf_values','rf_masks','rf_intents','geometry','context','candidates','mask','teacher','gain','chosen','prediction','prediction_supported']
    data,meta=read_episode(prefix,'validation',keys)
    if not meta['deterministic']:raise ValueError('Use a deterministic validation episode')
    maximum=0.;different=0;supported=0
    with patch('cooperative.world.ControlPredictor',side_effect=AssertionError('No LightGBM predictor allowed')),patch('torch.nn.GRU',side_effect=AssertionError('No GRU allowed')):
        c=StreamingController(checkpoint)
        for k in range(len(data['geometry'])):
            if k:
                for t in range((k-1)*6,k*6):c.observe((t+1)*.5,data['rf_values'][t],data['rf_masks'][t],data['rf_intents'][t])
            out=c.act(k*3.,data['context'][k],data['candidates'][k],data['mask'][k],data['teacher'][k],data['gain'][k],data['geometry'][k])
            different+=int(np.sum(out['actions']!=data['chosen'][k]))
            np.testing.assert_array_equal(out['supported'],data['prediction_supported'][k])
            keep=out['supported'];supported+=int(keep.sum())
            if keep.any():maximum=max(maximum,float(np.max(np.abs(out['probability'][keep]-data['prediction'][k][keep]))))
    result=dict(passed=different==0 and maximum<=5e-6,run=args.run,checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(checkpoint),episode=prefix.relative_to(ROOT).as_posix(),decisions=len(data['geometry']),
        supported_link_decisions=supported,action_differences=different,max_probability_error=maximum,
        lightgbm_constructor_blocked=True,gru_constructor_blocked=True,device='cpu',real_hardware_tested=False)
    atomic_json(ROOT/'results/transformer/verification/streaming_replay.json',result);print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
