"""완료한 연구의 배포 ZIP과 난수별 원시 학습 증거를 별도 묶음으로 생성."""

import argparse
import json
import zipfile
from pathlib import Path

import torch

from cooperative.io import validated_result
from cooperative.refine import STUDY,MODELS,atomic_json,protocol
from swarm_sim.regions import ROOT,sha256_file


def usable(path):
    return (path.is_file() and '__pycache__' not in path.parts and not any(x.startswith('.') for x in path.relative_to(ROOT).parts)
            and path.suffix not in ['.pyc','.nbc','.nbi','.lock','.tmp'] and '.tmp.' not in path.name)


def index(paths):
    return [dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in paths]


def write_zip(output,paths,manifest,top='',manifest_name='PACKAGE_CONTENTS.json'):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:
            relative=path.relative_to(ROOT).as_posix()
            archive.write(path,top+relative,compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        archive.writestr(top+manifest_name,json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('ZIP CRC 실패: '+bad)
    temporary.replace(output)
    result=dict(file=str(output.resolve()),bytes=output.stat().st_size,files=len(paths)+1,sha256=sha256_file(output))
    print(json.dumps(result),flush=True)
    return result


def evidence(seed,output):
    if seed not in [27101,27102,27103]:raise ValueError('공식 독립 학습 난수 필요')
    plan=protocol();paths=[]
    for mode in ['risk','no_risk']:
        name=f'{mode}_s{seed}';folder=MODELS/name
        final=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if final['iteration']!=8 or sum(h['episodes'] for h in final['history'])!=64:
            raise ValueError('완료된 64회 학습 필요: '+name)
        records=sorted((STUDY/'training'/name).glob('i[0-9][0-9]_e[0-9][0-9].json'))
        if len(records)!=64:raise ValueError('완료된 64개 학습 메타데이터 필요')
        for meta in records:
            row=json.loads(meta.read_text(encoding='utf-8'));prefix=meta.with_suffix('')
            spec={k:row[k] for k in ['region','direction','kind','seed']}
            expected=sha256_file(folder/f'iteration_{int(meta.stem[1:3]):02d}.pt')
            if validated_result(prefix,spec,expected,rollout=True) is None or row.get('refinement_sources')!=plan['source_sha256']:
                raise ValueError('원시 학습 기록 무결성 실패: '+str(prefix))
            paths.extend([meta,meta.with_suffix('.npz')])
        if len(list((STUDY/'training'/name).glob('i[0-9][0-9]_e[0-9][0-9].npz')))!=64:
            raise ValueError('완료된 64개 학습 NPZ 필요')
        paths.extend(p for p in folder.glob('*.pt') if usable(p))
        paths.append(STUDY/'training'/name/'history.json')
    paths=sorted(set(paths));manifest=dict(format='cooperative-raw-training-evidence-v2',training_seed=seed,
                full_missions=128,risk_modes=['risk','no_risk'],extract_into='project root; keep results/ and model/ paths',
                source_sha256=plan['source_sha256'],forecaster_sha256=plan['forecaster_sha256'],files=index(paths))
    return write_zip(output,paths,manifest,manifest_name=f'research_evidence/training_s{seed}.json')


def release(output):
    plan=protocol();publication=json.loads((STUDY/'published_selection.json').read_text(encoding='utf-8'))
    if publication['source_sha256']!=plan['source_sha256'] or publication['test_used_for_selection']:
        raise ValueError('완료된 v2 게시 기록 필요')
    paths=[]
    for name in ['cooperative','swarm_sim','usable','advance','upgrade','tests','maps','cache/usable','datasets','docs']:
        paths.extend(p for p in (ROOT/name).rglob('*') if usable(p))
    paths.extend(p for p in (ROOT/'configs').rglob('*') if usable(p) and '.preflight' not in p.name)
    paths.extend(p for p in (ROOT/'model').rglob('*') if usable(p) and 'preflight_training' not in p.parts)
    for path in ROOT.iterdir():
        if usable(path) and path.suffix in ['.py','.txt','.bat','.md','.json'] and path.name!='PACKAGE_CONTENTS.json':paths.append(path)
    paths.extend(p for p in (ROOT/'results/cooperative').glob('*.json') if usable(p))
    for path in STUDY.rglob('*'):
        if not usable(path) or any(x in path.parts for x in ['preflight_training','baseline_code','profile']):continue
        if 'training' in path.relative_to(STUDY).parts and path.suffix=='.npz':continue
        if path.name in ['progress.json','resume_progress.json','finalization_progress.json']:continue
        paths.append(path)
    # 과거 예측 설명서의 작은 집계·메타데이터는 보존하고, 대형 예측 학습 배열은 DATA_INDEX로 분리.
    for name in ['results/advance','results/usable']:
        paths.extend(p for p in (ROOT/name).rglob('*.json') if usable(p))
    paths=sorted(set(paths));manifest=dict(format='complete-integrated-simulator-refinement-v2',base_archive_required=False,
                selected=publication['selected'],frozen_selection='results/refinement/selection.json',
                counts=dict(training_full_missions=384,validation_and_seed_diagnostics=336,primary_test=288,official_demo_captures=6),
                raw_training_npz='separate capstone_training_s27101/27102/27103.zip; see DATA_INDEX.md',
                source_sha256=plan['source_sha256'],publication_sha256=sha256_file(STUDY/'published_selection.json'),files=index(paths))
    return write_zip(output,paths,manifest,top='drone_forecast_prediction_final/')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['release','evidence']);parser.add_argument('--seed',type=int)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.kind=='evidence':
        if args.seed is None:parser.error('evidence에는 --seed 필요')
        evidence(args.seed,args.out)
    else:release(args.out)


if __name__=='__main__':main()
