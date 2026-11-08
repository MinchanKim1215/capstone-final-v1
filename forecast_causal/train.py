"""Four measured-feature tree ablations and two matched GRU continuations."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,milestone

def main():
    verify()
    if not read(ARTIFACTS/'input_preflight.json')['passed']:raise ValueError('Causalinputpreflight required')
    from .model_preflight import main as model_check
    model_check()
    items=read(ARTIFACTS/'data_manifest.json')['records'];value=dict(created_at=now(),data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),experiment_plan_sha256=sha(ARTIFACTS/'experiment_plan.json'),
        path_uncertainty_sha256=sha(ARTIFACTS/'path_uncertainty.json'),training_queries=sum(r['queries'] for r in items if r['role']=='train'),uniform_training_queries=sum(r['uniform_queries'] for r in items if r['role']=='train'),
        event_bags=sum(r['event_bags'] for r in items if r['role']=='train'),source_pins={k:sha(ROOT/k) for k in ('tree_train.py','neural_train.py','model.py','inference.py','metrics.py','data.py','features.py','physics.py')})
    path=ARTIFACTS/'training_plan.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Training plan changed')
    else:write(path,value)
    from .tree_train import run as trees
    from .neural_train import run as neural
    tree_models=trees(items);neural_models=neural(items)
    write(ARTIFACTS/'trained_models.json',dict(models=tree_models+neural_models,tree_seed_fits=8,tree_binary_heads=16,neural_seed_fits=4,created_at=now()))
    status('stage3_experts_trained',next_action='선택45에서 후보를 정하고 보정45에서 모든 문턱을 고정한 뒤 새42비행을 생성한다.')

if __name__=='__main__':main()
