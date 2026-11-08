"""Compare the received release ZIP against the live experiment without editing it."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime

ops = Path(__file__).resolve().parent
project = ops.parent / 'drone_forecast_joint_training'
archive = Path('C:/Users/mckim/Downloads/capstone_transformer_training_20261029.zip')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


with zipfile.ZipFile(archive) as packed:
    member = next(name for name in packed.namelist() if name.endswith('/RELEASE_MANIFEST.json'))
    prefix = member[:-len('RELEASE_MANIFEST.json')]
    manifest = json.loads(packed.read(member))
    original_source = {}
    mismatches = []
    missing = []
    counts = dict(immutable_files=0, python_sources=0, configurations=0, frozen_models=0)
    for name, expected in manifest['immutable'].items():
        received = hashlib.sha256(packed.read(prefix+name)).hexdigest()
        if received != expected:
            raise ValueError('Release ZIP does not match its manifest: '+name)
        path = project/name
        if not path.exists():
            missing.append(name)
            continue
        current = sha(path)
        if current != received:
            mismatches.append(dict(path=name, release_sha256=received, current_sha256=current))
        counts['immutable_files'] += 1
        counts['python_sources'] += int(name.endswith('.py'))
        counts['configurations'] += int(name.startswith('configs/'))
        counts['frozen_models'] += int(name.startswith('model/') and name.endswith('.pt'))
        if name.endswith('.py'):
            original_source[name] = received
    optional = manifest['optional_evidence']
    optional_suffixes = sorted({Path(name).suffix for name in optional})
    progress = json.loads((project/'results/transformer/train/transformer_main/progress.json').read_text())
    best = json.loads((project/'results/transformer/train/transformer_main/best.json').read_text())
    result = dict(checked_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        archive=str(archive), archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
        release_format=manifest['format'], counts=counts, immutable_mismatches=mismatches,
        immutable_missing=missing, optional_archive_name='capstone_transformer_raw_evidence_20261029.zip',
        optional_manifest_files=len(optional), optional_manifest_suffixes=optional_suffixes,
        optional_expected_bytes=manifest['packaging'],
        source_sha256=original_source, current_committed_updates=progress['committed_updates'],
        current_training_flights=progress['training_episodes'], current_best_update=best['step'],
        screenshot_library_archives_accessed=False,
        scope='Exact byte comparison with the locally received ZIP, not an unobserved Library download.')
    destination = ops/'reports/release_comparison_20261029.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    text = ['**학습 코드와 전달 ZIP 비교**', '', result['checked_at'], '',
        f"현재 로컬 ZIP: {archive.name}, {result['archive_bytes']:,} bytes.",
        f"SHA256: {result['archive_sha256']}", '',
        f"v5 불변 파일 {counts['immutable_files']}개를 ZIP 원본·manifest·현재 폴더 사이에서 대조했다. 불일치 {len(mismatches)}개, 누락 {len(missing)}개다.",
        f"그 안의 Python 소스는 {counts['python_sources']}개, configs 항목은 {counts['configurations']}개, 고정 .pt 모델은 {counts['frozen_models']}개다.", '',
        '현재 학습 알고리즘은 전달된 v5와 같다. Windows CUDA 환경 설치, 새 지도학습 자료, PPO 학습 가중치와 평가 기록, 외부 진단 도구가 추가됐다. 가중치가 더 학습된 것과 코드가 새로운 알고리즘으로 바뀐 것은 구분해야 한다.', '',
        f"현재 완료 update {progress['committed_updates']}, PPO 비행 {progress['training_episodes']}회, 제어 기준 best {best['step']}이다.", '',
        f"OPTIONAL_EVIDENCE.md와 release manifest는 추가 ZIP을 {len(optional)}개의 {', '.join(optional_suffixes)} 원시 실행 기록으로 설명한다. manifest에는 Python 코드나 새 모델이 추가 기록으로 나열되어 있지 않다.", '',
        '스크린샷의 라이브러리 ZIP 자체는 확보하지 못했다. 따라서 그 두 파일이 로컬 ZIP 및 추가 기록과 바이트까지 동일한지는 아직 확정할 수 없다. 스크린샷의 첫 묶음 용량·폴더명과 로컬 패키지 구성은 부합한다.', '',
        '파일 수나 압축 용량이 큰 쪽을 더 발전된 알고리즘으로 판단하지 않는다. 추가 ZIP이 문서 설명과 같다면 실행 증거가 더 완전한 동일 버전이다.', '']
    (ops/'reports/release_comparison_20261029.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256']}, ensure_ascii=False))
    if mismatches or missing:
        raise SystemExit('Live experiment differs from original immutable release')
