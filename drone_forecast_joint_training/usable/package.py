"""실행 가능한 시연/코드와 무거운 전체 예측 캐시의 구분 보존."""
from pathlib import Path
import json,zipfile,shutil
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file

RESULT=ROOT/'results/usable';DEST=ROOT.parent/'deliverables'
HEAVY={'dense','runtime','alarm_calibration_scores','confirmation_scores'}


def main():
    DEST.mkdir(exist_ok=True)
    for name in ('Transformer_운용개선_결과.html','통신위험_전체임무_재생.html'):
        shutil.copy2(ROOT/'docs'/name,DEST/name)
    # 새 난수 확인은 학습/기존 평가 원시 기록과 별도 묶음.
    confirm=DEST/'drone_forecast_confirmation8.zip'
    with zipfile.ZipFile(confirm,'w',compression=zipfile.ZIP_STORED) as z:
        for f in sorted((ROOT/'datasets/confirmation8').rglob('*')):
            if f.is_file():z.write(f,'drone_forecast_usable/'+f.relative_to(ROOT).as_posix())
    cache=DEST/'drone_forecast_prediction_scores.zip'
    with zipfile.ZipFile(cache,'w',compression=zipfile.ZIP_STORED) as z:
        for f in sorted((RESULT/'runs').rglob('*.npz')):
            if any(part in HEAVY for part in f.parts):z.write(f,'drone_forecast_usable/'+f.relative_to(ROOT).as_posix())
    index_path=ROOT/'DATA_INDEX.md'
    original=index_path.read_text().split('\n## 추가 자료')[0].split('\n- drone_forecast_confirmation8.zip')[0]
    index_path.write_text(original+'\n\n## 추가 자료\n',encoding='utf-8')
    with index_path.open('a',encoding='utf-8') as f:
        for p,label in [(confirm,'추가 8개 새 난수 비행'),(cache,'전체 예측 점수 캐시 · 재계산 없이 세부 확인')]:
            f.write(f'\n- {p.name} · {label} · {p.stat().st_size/1024**2:.1f} MiB\n  SHA-256: `{sha256_file(p)}`\n')
    paths=[]
    for folder in ('swarm_sim','upgrade','usable','model','docs'):
        paths.extend(f for f in (ROOT/folder).rglob('*') if f.is_file() and not any(p=='__pycache__' for p in f.parts) and f.suffix not in ('.pyc','.nbc','.nbi') and f.name!='run_training.py')
    for name in ('README.md','DATA_INDEX.md','requirements-usable.txt','usable_forecast.py','00_환경설정.bat','01_예측기_재생.bat'):paths.append(ROOT/name)
    paths.extend((ROOT/'configs').glob('usable*.json'));paths.append(ROOT/'configs/channel_core_reference.json')
    protocol=read_json(ROOT/'configs/usable_protocol.json');paths.extend(ROOT/e['path'] for e in protocol['maps'])
    paths.extend((ROOT/'cache/usable').glob('*_visibility.npz'));paths.extend((ROOT/'cache/usable').glob('*_motion.npz'));paths.extend((ROOT/'cache/usable').glob('*_motion.json'))
    for name in ('__init__.py','test_usable.py','test_usable_predictor.py','test_usable_spatial.py','test_usable_alerts.py'):paths.append(ROOT/'tests'/name)
    # 최종 보고서에 필요한 집계 결과. 미완료 시드·옛 선정 기록·실행 로그는 묶음에서 제외.
    for f in RESULT.glob('*.json'):
        if f.name not in ('initial_three_variant_selection.json','four_variant_selection_before_auxiliary_control.json',
                          'spatial_training_inputs.json','test_spatial_input_audit.json'):paths.append(f)
    paths.extend(RESULT.glob('*.csv'));paths.append(RESULT/'final_tests.log')
    for folder in (RESULT/'runs').iterdir():
        if not folder.is_dir() or not (folder/'run.json').exists():continue
        paths.extend(f for f in folder.iterdir() if f.is_file() and f.suffix in ('.pt','.json','.npy','.npz'))
    paths.append(ROOT/'datasets/usable45/collection_manifest.json')
    for row in read_json(RESULT/'replay_summary.json'):
        folder=ROOT/'datasets/usable45/episodes'/row['id'];paths.extend(folder/name for name in ('observations.npz','truth.npz','spatial_observations.npz','episode.json'))
    paths=sorted(set(paths));missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    index={'format':'usable-delivery-v1','demo_records':3,'full_original_records':270,'new_confirmation_records':8,
           'full_data_separate':True,'selected_run':read_json(RESULT/'selection.json')['selected_variant']+'_s11',
           'files':{f.relative_to(ROOT).as_posix():{'bytes':f.stat().st_size,'sha256':sha256_file(f)} for f in paths}}
    write_json(ROOT/'PACKAGE_CONTENTS.json',index);paths.append(ROOT/'PACKAGE_CONTENTS.json')
    out=DEST/'drone_forecast_usable_final.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f in paths:z.write(f,'drone_forecast_usable/'+f.relative_to(ROOT).as_posix())
    result={'code':str(out),'code_bytes':out.stat().st_size,'code_sha256':sha256_file(out),
            'confirmation':str(confirm),'scores':str(cache),'files':len(paths)}
    write_json(ROOT.parent/'private/delivery_index.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
