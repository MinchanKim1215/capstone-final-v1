"""Copy only verified identical old-selection predictions into new cohorts."""
from pathlib import Path
import shutil
from . import ARTIFACTS,ROOT
from .contracts import read,sha,digest,OLD_NEURAL,OLD_TREE
from .evaluate import model_pin


def identity(model,rows,implementation=None):
    if implementation is None:implementation=dict(evaluate=sha(ROOT/'evaluate.py'),feature_cache=sha(ROOT/'feature_cache.py'),metrics=sha(ROOT/'metrics.py'))
    return digest(dict(model=model_pin(model),input=[(r['key'],r['array_sha256'],sha(Path(r['cache_path'])/'manifest.json')) for r in rows],
                       implementation=implementation))[:16]


def main():
    dataset=ARTIFACTS/'dataset.json';target_rows=[r for r in read(dataset)['rows'] if r['role']=='selection'];count=0
    for model in (str(OLD_NEURAL),str(OLD_TREE),str(ARTIFACTS/'models/same_data_20000/model.json')):
        tag=read(model)['tag'];pinned=model_pin(model);target=ARTIFACTS/'evaluation'/tag/identity(model,target_rows)/'selection';target.mkdir(parents=True,exist_ok=True)
        lookup={r['key']:r for r in target_rows}
        for path in (ARTIFACTS/'evaluation'/tag).glob('*/selection/summary.json'):
            summary=read(path)
            if summary['model_pin']!=pinned or sha(summary['dataset_path'])!=summary['dataset_sha256']:continue
            original=[r for r in read(summary['dataset_path'])['rows'] if r['role']=='selection']
            acceptable=[identity(model,original)]
            for snapshot in (ARTIFACTS/'source_snapshots').glob('evaluate*.py'):acceptable.append(identity(model,original,sha(snapshot)))
            if path.parent.parent.name not in acceptable:continue
            old={r['key']:r for r in original}
            for case in summary['cases']:
                key=case['key']
                if key not in lookup or lookup[key]['array_sha256']!=case['source_sha256'] or old[key]['cache_path']!=lookup[key]['cache_path']:continue
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('A previous prediction changed')
                destination=target/(key+'.npz')
                if not destination.exists():shutil.copyfile(case['predictions_path'],destination);count+=1
    print('Verified identical predictions reused:',count)


if __name__=='__main__':main()
