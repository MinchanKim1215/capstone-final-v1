"""재현 가능한 코드/결과와 큰 원시 기록을 분리해 배포합니다."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
import argparse
import json
import shutil

from swarm_sim.regions import ROOT, read_json, write_json, sha256_file

DELIVER = ROOT.parent/'deliverables'


def package_raw():
    manifest = read_json(ROOT/'datasets'/'upgrade45'/'collection_manifest.json')
    train_regions = sorted({r['region'] for r in manifest['records'] if r['split'] == 'train'})
    parts = [
        ('drone_training_records_A_20261025.zip', set(train_regions[:15])),
        ('drone_training_records_B_20261025.zip', set(train_regions[15:])),
        ('drone_validation_test_records_20261025.zip', {r['region'] for r in manifest['records'] if r['split'] != 'train'}),
    ]
    DELIVER.mkdir(parents=True, exist_ok=True)
    output = []
    for filename, regions in parts:
        path = DELIVER/filename
        records = [r for r in manifest['records'] if r['region'] in regions]
        with ZipFile(path, 'w', ZIP_STORED, allowZip64=True) as z:
            z.writestr(filename.replace('.zip', '_README.txt'),
                '코드 ZIP을 푼 프로젝트 폴더 안에 이 데이터 ZIP을 풀어 주십시오.\n'
                'datasets/upgrade45/episodes/ 경로가 합쳐져야 합니다.\n'
                '원시 기록 전체 사용 시 A, B, validation_test 세 묶음을 모두 풀어 주십시오.\n'
                f'이 묶음: {len(regions)}개 지도, {len(records)}개 실행\n'
                f'실험 설정 SHA256: {manifest["protocol_sha256"]}\n')
            for record in records:
                for name in ('observations.npz', 'truth.npz', 'episode.json'):
                    file = ROOT/record['path']/name
                    z.write(file, file.relative_to(ROOT).as_posix())
        with ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError('원시 기록 ZIP 검사 실패')
        item = {'file': filename, 'path': str(path), 'bytes': path.stat().st_size,
                'sha256': sha256_file(path), 'regions': sorted(regions), 'episodes': len(records)}
        output.append(item)
        print(json.dumps({'stage': 'raw_package', 'file': filename, 'MiB': round(item['bytes']/1024**2, 1)}, ensure_ascii=False), flush=True)
    write_json(DELIVER/'raw_package_inventory.json', {'parts': output, 'episodes': sum(r['episodes'] for r in output)})
    write_json(ROOT/'docs'/'data_archive_index.json', {'parts': [{k:v for k,v in p.items() if k != 'path'} for p in output],
               'episodes': sum(r['episodes'] for r in output),
               'unzip_inside_project_root': True})
    return output


def keep(path):
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if any(p in ('__pycache__', '.venv', '.venv_upgrade') for p in parts) or path.suffix in ('.pyc', '.tmp'):
        return False
    if parts[0] == 'datasets':
        # 큰 배열 캐시를 일부만 넣어 완료 파일이 있는 것처럼 보이지 않게 합니다.
        return relative.as_posix() in ('datasets/upgrade45/collection_manifest.json',
            'datasets/upgrade45/prepared/index_summary.json',
            'datasets/upgrade45/prepared/normalizer_train30.json',
            'datasets/upgrade45/prepared/normalizer_train3.json')
    if parts[0] == 'cache':
        return False  # 필요하면 같은 코드로 재생성합니다.
    if len(parts) == 1 and path.suffix == '.bat' and not path.name.startswith(('20_', '21_')):
        return False
    if parts[0] == 'docs' and ('qa' in parts or path.suffix == '.ttf'):
        return False
    if parts[:2] == ('results', 'upgrade') and path.name == 'final_processing.log':
        return False
    return True


def package_code():
    if not read_json(ROOT/'results'/'upgrade'/'dense_comparison.json')['complete']:
        raise ValueError('최종 시험을 먼저 완료하십시오.')
    report = ROOT/'docs'/'Transformer_개선실험_보고서.html'
    if not report.exists():
        raise ValueError('검증된 보고서를 먼저 생성하십시오.')
    DELIVER.mkdir(parents=True, exist_ok=True)
    if (DELIVER/'raw_package_inventory.json').exists():
        raw = read_json(DELIVER/'raw_package_inventory.json')
        write_json(ROOT/'docs'/'data_archive_index.json', {'parts': [{k:v for k,v in p.items() if k != 'path'} for p in raw['parts']],
                   'episodes': raw['episodes'], 'unzip_inside_project_root': True})
    shutil.copyfile(report, DELIVER/report.name)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and keep(p))
    inventory = {'project': ROOT.name, 'files': [{'path': p.relative_to(ROOT).as_posix(),
                  'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files],
                 'raw_data_in_separate_three_archives': True,
                 'source_protocol': read_json(ROOT/'configs'/'upgrade_protocol.json')['protocol_sha256']}
    filename = 'drone_transformer_code_results_20261025.zip'
    path = DELIVER/filename
    with ZipFile(path, 'w', ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for file in files:
            z.write(file, ROOT.name+'/'+file.relative_to(ROOT).as_posix())
        z.writestr(ROOT.name+'/PACKAGE_CONTENTS.json', json.dumps(inventory, ensure_ascii=False, indent=2))
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('코드/결과 ZIP 검사 실패')
    item = {'file': filename, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path),
            'files': len(files), 'html_report': str(DELIVER/report.name)}
    write_json(DELIVER/'code_package_inventory.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['raw', 'code', 'all'])
    args = parser.parse_args()
    if args.stage in ('raw','all'): package_raw()
    if args.stage in ('code','all'): package_code()
