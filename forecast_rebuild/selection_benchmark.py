"""Identical held-out selection trajectories for every candidate and baseline."""
import json
from . import ARTIFACTS
from .contracts import read,write,now
from .evaluate import evaluate


def main():
    sample=ARTIFACTS/'samples/expanded_core/manifest.json'
    dataset=ARTIFACTS/'benchmarks/expanded_dataset.json'
    if not dataset.exists():write(dataset,dict(rows=read(sample)['contract']['records'],scope='Spatial selection/calibration only; confirmation not opened'))
    tags=['expanded_lgbm_base','expanded_lgbm_energy','expanded_residual_base','expanded_residual_energy',
          'expanded_lgbm_planned_energy','expanded_residual_planned_energy']
    models=['baseline']+[str(ARTIFACTS/'models'/tag/'model.json') for tag in tags]
    results=[]
    for model in models:
        path=evaluate(model,'selection','cuda',dataset)
        results.append(str(path))
        write(ARTIFACTS/'benchmarks/selection_progress.json',dict(state='evaluating',complete=len(results),total=len(models),results=results,updated_at=now()))
    write(ARTIFACTS/'benchmarks/selection_results.json',dict(results=results,scope='Selection only',created_at=now()))
    write(ARTIFACTS/'benchmarks/selection_progress.json',dict(state='completed',complete=len(results),total=len(models),results=results,updated_at=now()))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
