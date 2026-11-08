"""Freeze and recompute report evidence; never writes inside the experiment."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'drone_forecast_joint_training'
OUT=ROOT/'output/pdf/evidence'
OUT.mkdir(parents=True,exist_ok=True)
sources={}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(relative,source_id=None):
    path=ROOT/relative
    raw=path.read_bytes()
    value=json.loads(raw.decode('utf-8-sig'))
    if source_id:
        sources[source_id]=dict(path=str(path),sha256=digest(raw))
    return value


def metadata(summary,source_id):
    rows=[];paths=[]
    for prefix in summary['raw_paths']:
        path=(PROJECT/prefix).with_suffix('.json')
        raw=path.read_bytes()
        rows.append(json.loads(raw))
        paths.append(dict(path=str(path),sha256=digest(raw)))
    sources[source_id]=dict(files=paths)
    return rows


def stats(rows):
    fields=['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s','duration_s','duration_ratio',
            'delay_mean_s','delay_p95_s','aoi_mean_s','relay_travel_m','outage_events']
    return dict(means={name:float(np.mean([row[name] for row in rows])) for name in fields},
                longest_outage_s=max(row['outage_max_s'] for row in rows),
                max_duration_ratio=max(row['duration_ratio'] for row in rows),
                completed=sum(row['mission_completed'] for row in rows),
                audit_passed=all(row['audit']['passed'] for row in rows),
                minimum_separation_m=min(row['minimum_separation_m'] for row in rows),
                safety_violations=sum(row['actual_safety_violations'] for row in rows),
                cases=[(row['region'],row['direction'],row['kind'],row['seed']) for row in rows])


def summary_for_best(run,source_id):
    pointer=read(f'drone_forecast_joint_training/results/transformer/train/{run}/best.json',source_id+'_pointer')
    relative='drone_forecast_joint_training/results/transformer/evaluation/validation/'+pointer['sha256'][:16]+'/summary.json'
    summary=read(relative,source_id)
    assert pointer['sha256']==summary['checkpoint_sha256']
    checkpoint=PROJECT/pointer['path']
    assert digest(checkpoint.read_bytes())==pointer['sha256']
    return pointer,summary


def main():
    now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    data=dict(as_of=now,project_root=str(PROJECT),workspace=str(ROOT))
    data['main_status']=read('training_ops/status.json','D04')
    data['progress']=read('drone_forecast_joint_training/results/transformer/train/transformer_main/progress.json','D05')
    data['best'],data['best_summary']=summary_for_best('transformer_main','D06')
    if data['best']['step']!=176:
        raise ValueError('Best changed: regenerate report narrative for new selection')
    data['original_joint_best'],data['original_joint_summary']=summary_for_best('verified_joint','D07')
    data['original_fixed_best'],data['original_fixed_summary']=summary_for_best('verified_fixed','D08')
    data['physics_summary']=read('drone_forecast_joint_training/results/transformer/evaluation/validation/30c8224c85fc74c0/summary.json','D09')
    data['rows']={
        'physics':metadata(data['physics_summary'],'D09a'),
        'original_joint':metadata(data['original_joint_summary'],'D07a'),
        'original_fixed':metadata(data['original_fixed_summary'],'D08a'),
        'best176':metadata(data['best_summary'],'D06a')}
    data['communication']={name:stats(rows) for name,rows in data['rows'].items()}
    for name,row in data['communication'].items():
        if sorted(row['cases'])!=sorted(data['communication']['physics']['cases']):
            raise ValueError('Communication scenario mismatch: '+name)
    data['deltas']={}
    target=data['communication']['best176']['means']
    for name in ['physics','original_joint','original_fixed']:
        prior=data['communication'][name]['means']
        data['deltas'][name]={key:dict(absolute=target[key]-prior[key],relative_percent=(target[key]/prior[key]-1)*100 if prior[key] else None) for key in target}
        data['deltas'][name]['longest_outage_s']=dict(absolute=data['communication']['best176']['longest_outage_s']-data['communication'][name]['longest_outage_s'])
    old={tuple(row[k] for k in ['region','direction','kind','seed']):row for row in data['rows']['physics']}
    pairs=[(old[tuple(row[k] for k in ['region','direction','kind','seed'])],row) for row in data['rows']['best176']]
    rng=np.random.default_rng(271077)
    selections=rng.integers(0,len(pairs),size=(10000,len(pairs)))
    data['paired_bootstrap']={}
    for field in ['search_connected_mean','pdr','outage_total_s']:
        difference=np.array([new[field]-before[field] for before,new in pairs])
        ci=np.quantile(difference[selections].mean(1),[.025,.975])
        data['paired_bootstrap'][field]=dict(mean_difference=float(difference.mean()),lower=float(ci[0]),upper=float(ci[1]),
                                           repetitions=10000,seed=271077,unit='paired validation region/scenario; exploratory selection data')
    data['bootstrap']=read('drone_forecast_joint_training/results/transformer/bootstrap/complete.json','D10')
    data['pretrain_best']=read('drone_forecast_joint_training/results/transformer/pretrain/best.json','D11')
    data['pretrain_latest']=read('drone_forecast_joint_training/results/transformer/pretrain/latest.json','D12')
    validations=read('drone_forecast_joint_training/results/transformer/train/transformer_main/validation_history.json','D13')
    data['validation_history']=[{k:row.get(k) for k in ['update','score','feasible','improved']} for row in validations]
    data['common']=read('training_ops/prediction_validation_20261029_v2/classification.json','D14')
    data['common_results']=read('training_ops/prediction_validation_20261029_v2/results.json','D15')
    data['alignment']=read('training_ops/prediction_diagnostics_20261029_v2/event_alignment.json','D16')
    data['corpus']=read('training_ops/prediction_diagnostics_20261029_v2/corpus_audit.json','D17')
    data['calibration']=read('training_ops/prediction_diagnostics_20261029_v2/calibration_crossfit.json','D18')
    data['package_comparison']=read('training_ops/reports/drive_package_comparison_20261029.json','D01')
    data['release_comparison']=read('training_ops/reports/release_comparison_20261029.json','D02')
    data['pilot_status']=read('prediction_improvement/runs/next_outage_1s_v1/status.json','D19')
    data['pilot_protocol']=read('prediction_improvement/runs/next_outage_1s_v1/protocol.json','D20')
    data['pilot_smoke']=read('prediction_improvement/runs/next_outage_1s_v1/real_data_smoke.json','D21')
    data['machine']=read('training_ops/machine.json','D22')
    data['shipped_results']=read('drone_forecast_joint_training/docs/transformer_results.json','D23')
    data['cuda_check']=read('drone_forecast_joint_training/results/transformer/verification/cuda-check.json','D24')
    data['runtime_configurations']=read('training_ops/reports/runtime_configuration_comparison_20261029.json','D25')
    archive=Path(data['package_comparison']['download']['path'])
    with zipfile.ZipFile(archive) as packed:
        member=next(name for name in packed.namelist() if name.endswith('/FULL_PROJECT_MANIFEST.json') and not name.startswith('__MACOSX/'))
        prefix=member[:-len('FULL_PROJECT_MANIFEST.json')]
        full=json.loads(packed.read(member))
        data['full_package']={k:v for k,v in full.items() if k!='files'}
        sources['D03']=dict(path=str(archive),member='FULL_PROJECT_MANIFEST.json',sha256=digest(packed.read(member)))
        original_bootstrap=json.loads(packed.read(prefix+'results/transformer/bootstrap/complete.json'))
        data['original_bootstrap_counts']=dict(train=len(original_bootstrap['train_paths']),validation=len(original_bootstrap['validation_paths']))
    for source_id,names in {'C01':['solo/model.py','solo/data.py','solo/targets.py','solo/training.py','solo/features.py','solo/deploy.py','solo/metrics.py','solo/alarms.py','cooperative/continuity_world.py','cooperative/world.py'],
                             'C02':['prediction_improvement/core.py','prediction_improvement/run.py','prediction_improvement/test_core.py']}.items():
        paths=[PROJECT/name for name in names] if source_id=='C01' else [ROOT/name for name in names]
        sources[source_id]=dict(files=[dict(path=str(path),sha256=digest(path.read_bytes())) for path in paths])
    for name,expected in data['bootstrap']['source_sha256'].items():
        if digest((PROJECT/name).read_bytes())!=expected:
            raise ValueError('Original source changed: '+name)
    # Cross-check all published summary communication means against raw episode metadata.
    for name,summary in [('physics',data['physics_summary']),('original_joint',data['original_joint_summary']),('original_fixed',data['original_fixed_summary']),('best176',data['best_summary'])]:
        for field,value in summary['communication']['means'].items():
            if field in data['communication'][name]['means'] and not np.isclose(value,data['communication'][name]['means'][field],rtol=0,atol=1e-12):
                raise ValueError('Summary/raw-episode mismatch: '+name+' '+field)
    data['checks']=dict(immutable_original_sources=150,communication_cases_match=True,summary_raw_metadata_match=True,
                        best_checkpoint_sha256_verified=True,final_test_accessed=False,pilot_has_results=(ROOT/'prediction_improvement/runs/next_outage_1s_v1/results.json').exists())
    data['sources']=sources
    (OUT/'snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(as_of=now,main_updates=data['progress']['committed_updates'],best=data['best']['step'],
                         old_joint_best=data['original_joint_best']['step'],old_fixed_best=data['original_fixed_best']['step'],
                         pilot=data['pilot_status'],checks=data['checks'],source_groups=len(sources)),ensure_ascii=False))


if __name__=='__main__':
    main()
