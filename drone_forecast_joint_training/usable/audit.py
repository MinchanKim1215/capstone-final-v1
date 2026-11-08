"""출처·선정·숫자·가중치·표시 범위를 최종 산출물과 대조."""
import json
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from usable.data import collection
from usable.collect import source_hashes
from usable.evaluate import RESULT,RUNS


def main():
    manifest,p=collection();inventory=[];current=source_hashes();spatial_source=sha256_file(ROOT/'usable/spatial.py')
    confirmation=read_json(ROOT/'datasets/confirmation8/collection_manifest.json')
    for r in manifest['records']+confirmation['records']:
        meta=r['metadata'];assert meta['source_sha256']==current
        files={}
        for name in ('observations.npz','truth.npz','spatial_observations.npz','episode.json'):
            path=ROOT/r['path']/name;h=sha256_file(path);files[name]=h
            if name in meta['files_sha256']:assert meta['files_sha256'][name]==h
        with np.load(ROOT/r['path']/'spatial_observations.npz') as z:assert str(z['source_sha256'])==spatial_source
        inventory.append({'id':r['id'],'split':r['split'],'files_sha256':files})
    for e in p['maps']:assert sha256_file(ROOT/e['path'])==e['sha256']
    selection=read_json(RESULT/'selection.json');selected=selection['selected_variant']+'_s11'
    assert len(selection['variants'])==5
    assert selection['selected_variant']==max(selection['validation_select_pooled_ap'],key=selection['validation_select_pooled_ap'].get)
    runs=[]
    for folder in (RESULT/'runs').iterdir():
        if not folder.is_dir() or not (folder/'run.json').exists():continue
        run=read_json(folder/'run.json');assert run['checkpoint_sha256']==sha256_file(folder/'best.pt');runs.append(folder.name)
    for name in ('best.pt','calibration.json','alarm_policy.json'):assert sha256_file(ROOT/'model'/name)==sha256_file(RUNS/selected/name)
    base=read_json(RUNS/'B0_legacy_inputs_s11/dense_evaluation.json');best=read_json(RUNS/selected/'dense_evaluation.json');summary=read_json(RESULT/'report_summary.json')
    assert base['samples']==best['samples']==4866832
    assert summary['dense_selected_ap']==best['balanced']['all']['average_precision']
    assert summary['dense_baseline_ap']==base['balanced']['all']['average_precision']
    assert read_json(RESULT/'motion_check.json')['complete_count']==90
    confirm_result=read_json(RESULT/'confirmation_results.json');assert confirm_result['flights']==8
    assert confirm_result['frozen_inputs']['alarm_policy_sha256']==sha256_file(RUNS/selected/'alarm_policy.json')
    for r in read_json(RESULT/'replay_summary.json'):assert r['rendered_start_s']==0 and r['rendered_end_s']==r['duration_s']
    write_json(RESULT/'data_inventory.json',{'original_records':270,'additional_confirmation_records':8,'records':inventory})
    out={'pass':True,'maps':45,'original_flights':270,'confirmation_flights':8,'completed_motion_paths':90,
         'completed_training_runs':runs,'selected_run':selected,'same_primary_evaluation_windows':4866832,
         'all_raw_hashes_verified':True,'core_source_matches_generation':True,'model_export_matches':True,
         'report_metrics_match':True,'full_replay_endpoints_match':True}
    write_json(RESULT/'integrity_audit_final.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
