"""Short new-map weight-transfer/update checks; never starts a campaign."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import adaptive_joint_expanded
import torch
from adaptive_joint_expanded import ARTIFACTS
from adaptive_joint_expanded.contracts import read_json,atomic_json,plan,sha,seed_all,DEPLOYMENTS
from adaptive_joint_expanded.experiment import assert_protocol,save_checkpoint,collect
from adaptive_joint_expanded.model import Controller,optimizer
from adaptive_joint_expanded.learning import update


def main():
    p=read_json(ARTIFACTS/'protocol.json');assert_protocol(p)
    c=dict(p['config'],device='cpu',epochs=1)
    source=ROOT/'adaptive_joint/artifacts/runs/adaptive_v6_20261030/checkpoints/s0000144.pt'
    assert sha(source)=='735909206e4c701db8ead31455cdbae9a34b4daaf0c0fa4904da9b8e30f2868a'
    ck=torch.load(source,map_location='cpu',weights_only=False)
    seed_all(c['seed']);model=Controller(warm_start=False);model.load_state_dict(ck['model'],strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in ck['model'].items())
    folder=ARTIFACTS/'expanded_probe';opt=optimizer(model,c)
    initial=save_checkpoint(folder/'initial',model,opt,c,0,p,[],purpose='preflight_only')
    maps=[m for m in plan()['maps'] if m.get('source_dataset','legacy_v6')!='legacy_v6' and m['joint_split']=='train']
    chosen=[next(m for m in maps if m['province']==province)['key'] for province in ('gyeonggi','incheon')]
    specs=[dict(region=k,direction='east_west',kind='intermittent',seed=2026103105) for k in chosen for _ in DEPLOYMENTS]
    print(json.dumps(dict(stage='new_map_short_rollouts',regions=chosen)),flush=True)
    episodes=collect(initial,p,specs,folder/'short', [d.key for d in DEPLOYMENTS]*2,False,24)
    stats=update(model,opt,episodes,c,1,'cpu')
    assert stats['gradient_proof']['encoder_from_policy']>0 and stats['gradient_proof']['timing_from_policy']>0
    assert stats['weight_change_l2']['actor']>0 and stats['weight_change_l2']['forecaster.timing']>0
    atomic_json(folder/'short_checks.json',dict(passed=True,purpose='Preflight only; not performance evidence',
                source_checkpoint_sha256=sha(source),all_parent_tensors_transferred=True,
                statistics=stats,regions=chosen,all_safety_passed=all(r['audit']['passed'] for _,r in episodes)))
    print(json.dumps(dict(stage='new_map_actual_update_passed',samples=stats['samples'])),flush=True)
    full_specs=[dict(region=chosen[1],direction='north_south',kind='intermittent',seed=2026103110) for _ in DEPLOYMENTS]
    rows=collect(initial,p,full_specs,folder/'full_four',[d.key for d in DEPLOYMENTS],True,None,load_arrays=False)
    assert all(r['mission_completed'] and r['audit']['passed'] for r in rows)
    atomic_json(folder/'full_checks.json',dict(passed=True,purpose='New-map execution only; not performance evidence',
                rows=[dict(deployment=r['deployment'],completed=r['mission_completed'],audit=r['audit']) for r in rows]))
    print(json.dumps(dict(stage='new_map_full_four_passed',region=chosen[1])),flush=True)


if __name__=='__main__':main()
