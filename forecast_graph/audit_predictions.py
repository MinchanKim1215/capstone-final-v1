"""Independently recalculate small batches from every candidate and data fold."""
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .evaluate import physical,previous,prediction_path
from .inference import predict


def main():
    frozen=read(ARTIFACTS/'freeze.json');done=read(ARTIFACTS/'completed_comparison.json')
    summaries=[]
    for item in frozen['candidates']:summaries.extend([item['selection_summary'],item['calibration_summary']])
    summaries.extend(item['raw_summary'] for item in done['outcomes'])
    files=0
    for path in summaries:
        summary=read(path)
        if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Evaluated dataset changed')
        for case in summary['cases']:
            if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Stored evaluated probabilities changed')
            files+=1
    data=read(ARTIFACTS/'data_manifest.json')['records']+read(ARTIFACTS/'confirmation_data_manifest.json')['records']
    sampled=[]
    for scope in ('selection','calibration','confirmation'):
        for kind in ('clear','continuous','intermittent'):
            sampled.append(next(i for i in data if i['role']==scope and i['source_record']['spec']['kind']==kind))
    models={c['candidate']['tag']:c['candidate']['model'] for c in frozen['candidates'] if c['candidate']['task']=='link' and c['candidate']['family']=='new'}
    maxima={tag:0. for tag in list(models)+['previous_selected','v11_hybrid_blend50']};counts={tag:0 for tag in maxima}
    for index,item in enumerate(sampled):
        record=item['source_record'];ep=physical(record)
        with np.load(item['path'],allow_pickle=False) as a:
            positions=np.linspace(0,len(a['n'])-1,3).astype(int);arrays={key:a[key][positions] for key in a.files}
        n=ep.values.shape[2];expected={}
        for tag,model in models.items():
            result=predict(model,arrays,record,ep,device='cuda')
            expected[tag]=dict(link=result['link'][:,:n],service=result['service'])
        expected['previous_selected']={task:previous(task,item,arrays,ep,'confirmation') for task in ('link','service')}
        expected['v11_hybrid_blend50']={task:.5*expected['previous_selected'][task]+.5*expected['v11_hybrid_shared'][task] for task in ('link','service')}
        t=arrays['query'][:,0]
        for tag,values in expected.items():
            for task,value in values.items():
                candidate=next(i['candidate'] for i in frozen['candidates'] if i['candidate']['task']==task and i['candidate']['tag']==tag)
                with np.load(prediction_path(candidate,record['key']),allow_pickle=False) as a:p=a['prediction']
                actual=p[t,0] if task=='link' else p[t,0,0]
                good=np.isfinite(actual)
                if not np.isfinite(value[good]).all():raise ValueError('Recomputed prediction lost an evaluated output')
                if good.any():
                    error=float(np.abs(actual[good]-value[good]).max());maxima[tag]=max(maxima[tag],error);counts[tag]+=int(good.sum())
                    np.testing.assert_allclose(actual[good],value[good],atol=1e-4,rtol=1e-4)
        write(ARTIFACTS/'audit_prediction_status.json',dict(complete=index+1,total=len(sampled),updated_at=now()))
    value=dict(passed=True,created_at=now(),prediction_files_hashed=files,independent_flights=len(sampled),
        finite_predictions_checked=counts,maximum_raw_probability_difference=maxima,
        note='Same source arrays and frozen weights. Float32 batch-size/backend differences bounded; stored forecasts and policies unchanged.')
    write(ARTIFACTS/'audit_predictions.json',value);return value

