"""v3 전체 소스·새 원시 학습/검증/시험을 한 ZIP으로 보존한다."""
import argparse
import json
from pathlib import Path

from cooperative.continuity import STUDY, protocol, verify_selection
from cooperative.refine import atomic_json
from cooperative.package_release import usable, index, write_zip
from swarm_sim.regions import ROOT, sha256_file


def paths_for_package():
    paths=[]
    def acceptable(p):
        return usable(p) and not any(x.startswith('preflight_') or x in ['baseline_code','profile'] for x in p.relative_to(ROOT).parts)
    for name in ['cooperative','swarm_sim','usable','advance','upgrade','tests','maps',
                 'cache/usable','datasets','docs','configs','model','results/refinement','results/continuity','results/deployment']:
        for p in (ROOT/name).rglob('*'):
            if not acceptable(p) or p.name in ['progress.json','resume_progress.json','finalization_progress.json']:continue
            if p.suffix=='.npz' and p.relative_to(ROOT).parts[:2]==('results','refinement'):continue
            if 'results' in p.relative_to(ROOT).parts and p.suffix=='.npz':
                # No in-progress array without its final atomic metadata sidecar.
                if not p.with_suffix('.json').exists():continue
            paths.append(p)
    for p in ROOT.iterdir():
        if acceptable(p) and p.suffix in ['.py','.txt','.bat','.md','.json'] and p.name!='PACKAGE_CONTENTS.json':paths.append(p)
    paths.extend(p for p in (ROOT/'results/cooperative').glob('*.json') if acceptable(p))
    for name in ['results/advance','results/usable']:
        paths.extend(p for p in (ROOT/name).rglob('*.json') if acceptable(p))
    return sorted(set(paths))


def package(output,checkpoint=False):
    core=protocol()
    deployment=(ROOT/'configs/deployment_plan.json').exists()
    study=ROOT/'results/deployment' if deployment else STUDY
    if deployment:
        from cooperative.deployment import plan as deployment_plan
        frozen=deployment_plan()['source_sha256']
    else:frozen=core['source_sha256']
    if not checkpoint:
        if deployment:
            from cooperative.deployment_publish import verify_publication
            pub=verify_publication();audited=json.loads((study/'audited_summary.json').read_text(encoding='utf-8'))
        else:
            freeze=verify_selection(core)
            pub=json.loads((study/'published_selection.json').read_text(encoding='utf-8'))
            audited=json.loads((study/'audited_summary.json').read_text(encoding='utf-8'))
            if pub['selected']!=freeze['selected'] or not audited['formal_safety']['all_passed']:raise ValueError('Complete v3 release required')
    paths=paths_for_package()
    source={p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths if p.suffix in ['.py','.html'] and p.relative_to(ROOT).parts[0] in ['cooperative','tests']}
    source['cooperate_run.py']=sha256_file(ROOT/'cooperate_run.py')
    manifest=dict(format=('communications-deployment-checkpoint-v3.1' if deployment else 'communications-priority-checkpoint-v3') if checkpoint else ('complete-integrated-simulator-deployment-v3.1' if deployment else 'complete-integrated-simulator-continuity-v3'),
        base_archive_required=False,includes_new_raw_training=True,
        current_study=str(study.relative_to(ROOT)),
        v2_raw_data='optional earlier v2 release version and training_s27101/27102/27103 archives; not required for current study',
        omitted_historical_npz='results/refinement v2 validation/test/demo arrays only; source,models,metadata preserved',
        frozen_source_sha256=frozen,package_source_sha256=source,files=index(paths))
    if checkpoint:
        manifest['checkpoint_progress']=json.loads((study/'progress.json').read_text(encoding='utf-8')) if (study/'progress.json').exists() else {'stage':'pilot'}
        manifest['included_complete_arrays']={phase:sum(1 for p in paths if p.suffix=='.npz' and (study/phase) in p.parents) for phase in ['development_validation','development_test','confirmation','pilot','training','validation','test']}
        manifest['resume_command']='python cooperate_run.py deployment --stage all --workers 4' if deployment else 'python cooperate_run.py continuity --stage all --workers 4'
    else:
        manifest.update(selected=pub['selected'],counts=audited['counts'],publication_sha256=sha256_file(study/'published_selection.json'),selection_sha256=sha256_file(study/'selection.json'))
    return write_zip(output,paths,manifest,top='drone_forecast_prediction_final/')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',action='store_true');a=p.parse_args();package(a.out,a.checkpoint)
