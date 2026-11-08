"""Only map-level training archives can enter supervised optimization."""
from pathlib import Path
from . import WORKSPACE, ARTIFACTS
from .contracts import read,write,sha,prepare,maps,role


def build():
    prepare();registry=maps();rows=[];seen=set()
    pilot=WORKSPACE/'prediction_improvement/runs/next_outage_1s_v1/data'
    for meta in sorted(pilot.glob('*.json'),key=lambda p:int(p.stem)):
        item=read(meta);spec=item['spec'];region=spec['region']
        if registry.get(region,{}).get('joint_split')!='train':continue
        source=WORKSPACE/'drone_forecast_joint_training'/(spec['prefix']+'.npz')
        if not source.exists():raise FileNotFoundError(source)
        raw=read(source.with_suffix('.json'))
        record=dict(role='train',region=region,source_kind='bootstrap',
            metadata_path=str(source.with_suffix('.json')),array_path=str(source),
            array_sha256=spec['archive_sha256'],geometry_path=str(meta.with_suffix('.npz')),
            geometry_sha256=item['sha256'],n_search=3,deployment='v8_s3_r5_40',
            spec={k:raw[k] for k in ('region','direction','kind','seed')})
        rows.append(record);seen.add(str(source.resolve()))
    for package,run in [('adaptive_joint','adaptive_v6_20261030'),('adaptive_joint_expanded','adaptive_v7_expanded100_20261031')]:
        folder=WORKSPACE/package/'artifacts/runs'/run/'rollouts'
        for source in sorted(folder.glob('u*/*.npz')):
            item=read(source.with_suffix('.json'));region=item['region']
            if registry.get(region,{}).get('joint_split')!='train':continue
            if str(source.resolve()) in seen:continue
            rows.append(dict(role='train',region=region,source_kind='joint',metadata_path=str(source.with_suffix('.json')),
                array_path=str(source),array_sha256=item['array_sha256'],n_search=item['search_drones'],
                deployment=item['deployment'],spec=item['spec']))
            seen.add(str(source.resolve()))
    # Historical development data can support a preliminary engineering run.
    # They never become training examples. Confirmation remains unread until
    # the new study's model/alarm freeze, even if older outcomes were seen.
    for package,run,step in [('adaptive_joint','adaptive_v6_20261030',192),
                             ('adaptive_joint_expanded','adaptive_v7_expanded100_20261031',64)]:
        folder=WORKSPACE/package/'artifacts/runs'/run/'validation'/f'u{step:07d}'/'rollouts'
        for source in sorted(folder.glob('*.npz')):
            item=read(source.with_suffix('.json'));region=item['region']
            if registry.get(region,{}).get('joint_split')!='validation':raise ValueError('Unexpected historical development partition')
            if str(source.resolve()) in seen:continue
            rows.append(dict(role=role(region),region=region,source_kind='joint_development',
                metadata_path=str(source.with_suffix('.json')),array_path=str(source),array_sha256=item['array_sha256'],
                n_search=item['search_drones'],deployment=item['deployment'],spec=item['spec']))
            seen.add(str(source.resolve()))
    for fresh in [ARTIFACTS/'collection_manifest.json',ARTIFACTS/'balanced_collection_manifest.json']:
        if not fresh.exists():continue
        for row in read(fresh)['rows']:
            source=Path(row['array_path'])
            if str(source.resolve()) in seen:continue
            item=read(row['metadata_path']);expected=role(row['region'])
            if row['role']!=expected:raise ValueError('Fresh archive partition mismatch')
            rows.append(dict(row,source_kind='fresh_v7',n_search=item['search_drones'],deployment=item['deployment'],spec=item['spec']))
            seen.add(str(source.resolve()))
    for i,row in enumerate(rows):
        row['key']=sha(Path(row['metadata_path']))[:20]
        if row['role']=='train' and registry[row['region']]['joint_split']!='train':
            raise ValueError('Development geography in training corpus')
    result=dict(rows=rows,counts={key:sum(r['role']==key for r in rows) for key in ('train','selection','calibration','confirmation')},
                confirmation_use='Collection only until model and alarm freeze')
    write(ARTIFACTS/'inventory.json',result)
    return result


if __name__=='__main__':
    import json
    item=build();print(json.dumps(item['counts']))
