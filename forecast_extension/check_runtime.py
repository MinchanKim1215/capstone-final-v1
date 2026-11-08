"""Whole-flight comparison of frozen batch predictions and local stream."""
import numpy as np
from pathlib import Path
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .evaluate import verify_freeze
from .runtime import Forecast,runtime_dependencies
from .metrics import alert_counts
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context
from forecast_rebuild.inference import Predictor
from forecast_rebuild.calibration import transform,alarm_support


def main():
    dependencies=runtime_dependencies();frozen=verify_freeze();item=next(i for i in frozen['models'] if i['pin']['path']==frozen['selected_model'])
    result_path=ARTIFACTS/'runtime_checks.json'
    contract=dict(freeze_sha256=sha(ARTIFACTS/'freeze.json'),checker_sha256=sha(__file__),
                  runtime_sha256=sha(Path(__file__).with_name('runtime.py')),runtime_dependencies=dependencies)
    if result_path.exists():
        previous_check=read(result_path)
        if all(previous_check.get(key)==value for key,value in contract.items()):
            print('Reusing the identical frozen-model whole-flight check:',result_path);return
    summary=read(item['selection_summary']);records=read(summary['dataset_path'])['rows']
    record=next(r for r in records if r['role']=='selection' and r['source_kind']=='fresh_v9');episode=Episode(record)
    city,nominal=public_context(record);n=episode.values.shape[2];ns=record['n_search'];frames=len(episode.values)-4;frames-=frames%2
    stream=Forecast(frozen['selected_model'],city,ns,n,ns,policy=item['policy'],nominal=nominal,max_frames=len(episode.values)+2)
    model=Predictor(frozen['selected_model'],'cpu');policy=read(item['policy']);previous=np.zeros(n,bool)
    tested=0;maximum=0.;feature_max=0.;mismatches=0;latency=[]
    support=alarm_support(episode,policy['require_energy'])
    for q in range(frames):
        result=stream.observe((q+1)*.5,episode.values[q,0],episode.masks[q,0],episode.intents[q,0],episode.energy[q,0])
        if result['status']!='ready':continue
        t=int(np.flatnonzero(episode.ends==q+1)[0]);peers=[j for j in range(n) if j!=ns]
        rows=np.array([[t,0,j] for j in peers]);pred=transform(model.predict(episode,rows)[0],policy['parameters'])
        if model.meta['family']!='baseline':
            a=episode.features(rows,model.meta.get('use_energy',False),model.meta.get('use_plan',False))
            b=stream.features(np.array([[0,0,j] for j in peers]),model.meta.get('use_energy',False),model.meta.get('use_plan',False))
            np.testing.assert_equal(np.isnan(a),np.isnan(b));feature_max=max(feature_max,float(np.max(np.abs(np.nan_to_num(a)-np.nan_to_num(b)))))
        for k,row in enumerate(result['links']):
            j=row['peer']
            if row['risk_5s'] is not None:
                maximum=max(maximum,float(abs(row['risk_5s']-pred[k,2])),float(abs(row['risk_3to7s']-pred[k,5])));tested+=1
            call=bool(support[t,0,j] and pred[k,5]>=(policy['threshold']*.5 if previous[j] else policy['threshold']))
            mismatches+=int(row['qualified_first_alert']!=(call and not previous[j]));previous[j]=call
            assert not row['trusted_alarm'] and row['alarm_data_eligible']==bool(support[t,0,j])
        latency.append(result['inference_ms'])
    assert tested>0 and maximum<2e-5 and feature_max<2e-6 and mismatches==0,(tested,maximum,feature_max,mismatches)
    for q in (frames,frames+1):
        mask=np.array(episode.masks[q,0]);mask[:,41]=False
        result=stream.observe((q+1)*.5,episode.values[q,0],mask,episode.intents[q,0])
    assert all(row['risk_5s'] is None and not row['trusted_alarm'] for row in result['links'])
    out=dict(created_at=now(),selected_tag=item['tag'],tested_record_key=record['key'],raw_source_sha256=record['array_sha256'],
        frames=frames,tested_supported_link_predictions=tested,
        max_abs_probability_error=maximum,max_abs_feature_error=feature_max,alarm_decision_mismatches=mismatches,
        missing_RF_returns_None=True,trusted_alarm_disabled=True,runtime_sha256=sha(Path(__file__).with_name('runtime.py')),
        runtime_dependencies=dependencies,
        cpu_ms=dict(cold=latency[0],warm_median=float(np.median(latency[1:])),warm_max=float(np.max(latency[1:]))),hardware_latency_certified=False)
    out.update(contract);write(result_path,out);print(out)


if __name__=='__main__':
    main()
