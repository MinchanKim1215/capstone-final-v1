"""Finalize already committed RF archives after metadata-only interruption."""
from pathlib import Path
import json
from . import ARTIFACTS
from .contracts import prepare,read,write,sha,now,role


def main():
    protocol=prepare();plan=read(ARTIFACTS/'collection_plan.json');rows=[]
    folder=ARTIFACTS/'raw/current'
    for i,(spec,forced,split) in enumerate(zip(plan['specs'],plan['forced'],plan['roles'])):
        prefix=folder/f'{i:04d}_{spec["region"]}_{forced or "adaptive"}'
        meta=prefix.with_suffix('.json');path=prefix.with_suffix('.npz');item=read(meta)
        if role(spec['region'])!=split:raise ValueError('Spatial split mismatch')
        if item['request']['spec']!=spec or item['request']['checkpoint_sha256']!=protocol['baseline']['sha256']:
            raise ValueError('Committed trajectory request differs')
        if sha(path)!=item['array_sha256']:raise ValueError('Committed trajectory hash differs')
        rows.append(dict(role=split,metadata_path=str(meta),array_path=str(path),array_sha256=item['array_sha256'],region=spec['region']))
    write(ARTIFACTS/'collection_manifest.json',dict(rows=rows,verified_at=now(),recovered_after_metadata_only_interruption=True))
    write(ARTIFACTS/'collection_status.json',dict(state='completed',completed=len(rows),total=len(plan['specs']),ended_at=now(),
        recovered_after_metadata_only_interruption=True,all_archive_hashes_verified=True,original_models_modified=False,
        final_test_opened=False,new_predictor_training_started=False))
    print(json.dumps(dict(state='completed',verified_archives=len(rows))))


if __name__=='__main__':main()
