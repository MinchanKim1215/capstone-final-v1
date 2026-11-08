"""Correct a copied pre-confirmation count header without changing outcomes.

The frozen builder appends confirmation rows but retains its old count header.
Keep the executed input and summaries, and explicitly rebind only this metadata
after every fixed candidate has finished. No predictions or metrics are refit.
"""
from collections import Counter
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now


def correct_confirmation_counts(completed):
    folder=ARTIFACTS/'metadata_correction';plan_path=folder/'record_counts.json'
    dataset_path=ARTIFACTS/'dataset_with_confirmation.json'
    if not plan_path.exists():
        original=read(dataset_path);counts=dict(Counter(row['role'] for row in original['rows']))
        if original['counts']==counts:return dict(required=False)
        if original['counts']!=read(ARTIFACTS/'dataset.json')['counts']:raise ValueError('Unexpected count-header mismatch')
        folder.mkdir(exist_ok=True);write(folder/'dataset_original.json',original)
        corrected=dict(original,counts=counts,pre_confirmation_counts=original['counts'],
            count_metadata_note='Post-evaluation correction of the copied count header only; all row records and outcomes unchanged.')
        write(folder/'dataset_corrected.json',corrected);summaries=[]
        for outcome in completed['outcomes']:
            summary=read(outcome['raw_summary'])
            if summary['dataset_sha256']!=sha(dataset_path):raise ValueError('Original summary input pin differs')
            snapshot=folder/(outcome['tag']+'_summary_original.json');write(snapshot,summary)
            summaries.append(dict(path=outcome['raw_summary'],snapshot=str(snapshot)))
        write(plan_path,dict(state='prepared',created_at=now(),original_dataset_sha256=sha(dataset_path),
            corrected_dataset_sha256=sha(folder/'dataset_corrected.json'),actual_counts=counts,summaries=summaries))
    plan=read(plan_path)
    if sha(dataset_path) not in (plan['original_dataset_sha256'],plan['corrected_dataset_sha256']):raise ValueError('Dataset changed beyond the documented count correction')
    original=read(folder/'dataset_original.json');corrected=read(folder/'dataset_corrected.json')
    if digest(original['rows'])!=digest(corrected['rows']):raise ValueError('Count correction changed an input record')
    write(dataset_path,corrected);evidence=[]
    for item in plan['summaries']:
        original_summary=read(item['snapshot']);current=read(item['path'])
        def substantive(value):return {key:val for key,val in value.items() if key not in ('dataset_sha256','dataset_metadata_correction')}
        if digest(substantive(current))!=digest(substantive(original_summary)):raise ValueError('A comparison outcome changed')
        changed=dict(original_summary,dataset_sha256=plan['corrected_dataset_sha256'],dataset_metadata_correction=dict(
            original_dataset_sha256=plan['original_dataset_sha256'],record_counts_only=True,all_input_records_and_metrics_unchanged=True))
        write(item['path'],changed)
        evidence.append(dict(path=item['path'],original_sha256=sha(item['snapshot']),corrected_sha256=sha(item['path']),
            identical_substantive_content_sha256=digest(substantive(changed))))
    plan.update(state='completed',completed_at=now(),input_record_content_unchanged=True,all_metric_content_unchanged=True,evidence=evidence)
    write(plan_path,plan);return dict(required=True,record=str(plan_path),actual_counts=plan['actual_counts'],metrics_unchanged=True)
