"""Resolve archived PC paths while retaining the frozen source/evidence bytes."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
PATHS = json.loads((ROOT / 'handoff_paths.json').read_text(encoding='utf-8'))
OLD_ROOT = PATHS['source_root'].replace('\\', '/').rstrip('/')


def resolve(path):
    value = str(path).replace('\\', '/')
    current_release = ROOT.parent.as_posix()
    if value.casefold() == current_release.casefold() or value.casefold().startswith(current_release.casefold() + '/'):
        return Path(path)
    if value.casefold().startswith(OLD_ROOT.casefold() + '/'):
        return ROOT / value[len(OLD_ROOT) + 1:]
    current = Path(path)
    return current if current.is_absolute() else ROOT / current


def read_evidence(path):
    data = json.loads(resolve(path).read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'archive_path' in data:
        data['archive_path'] = str(resolve(data['archive_path']))
    return data


def sha(path):
    digest = hashlib.sha256()
    with resolve(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def activate():
    legacy = ROOT / 'drone_forecast_joint_training'
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    from adaptive_joint_expanded import contracts
    from forecast_learning import control, training
    from final_v1 import evaluation
    from final_v1 import training as joint_training

    original_plan = contracts.plan

    def relocated_plan():
        plan = original_plan()
        for entry in plan['maps']:
            path = Path(entry['path'])
            raw = str(path).replace('\\', '/')
            source = raw if raw.casefold().startswith(OLD_ROOT.casefold() + '/') else legacy / path
            entry['path'] = str(resolve(source))
        return plan

    def read_profile(path):
        data = read_evidence(path)
        if resolve(path) == control.ARTIFACTS / 'freeze.json':
            data = copy.deepcopy(data)
            for metadata in data['models'].values():
                metadata['path'] = str(resolve(metadata['path']))
        return data

    original_checkpoint = joint_training.read_checkpoint

    def read_checkpoint(path):
        return original_checkpoint(resolve(path))

    contracts.plan = relocated_plan
    training.read = read_evidence
    training.sha = sha
    control.read = read_profile
    control.load_profile.cache_clear()
    joint_training.read_checkpoint = read_checkpoint
    evaluation.read_checkpoint = read_checkpoint


def verify_files():
    release_root = ROOT.parent
    manifest = json.loads((release_root / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    failures = []
    for name, expected in manifest.items():
        path = release_root / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            failures.append(name)
            continue
        if sha(path) != expected['sha256']:
            failures.append(name)
    if failures:
        raise ValueError('Changed or missing delivery files: ' + ', '.join(failures[:12]))
    return dict(files_verified=len(manifest), maps=len(PATHS['maps']), checksums='passed')
