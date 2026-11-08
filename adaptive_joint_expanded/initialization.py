"""A new-data study may inherit weights only from a completed pinned parent."""
from pathlib import Path
from . import ROOT
from .contracts import read_json,sha,digest

BINDING=ROOT/'configs/parent_initialization.json'
PARENT=ROOT.parent/'adaptive_joint'
PARENT_RUN=PARENT/'artifacts/runs/adaptive_v6_20261030'


def parent_finished(status):
    return status.get('state') in ('completed','early_stopped') and status.get('stage')=='finished'


def parent_initialization(required=False):
    if not BINDING.exists():
        if required:raise ValueError('Parent study is not finished and bound; no cold restart is allowed')
        return None
    value=read_json(BINDING)
    if not parent_finished(read_json(PARENT_RUN/'status.json')):
        raise ValueError('Parent must finish before expanded training')
    p=read_json(PARENT/'artifacts/protocol.json')
    if value['protocol_sha256']!=digest(p):raise ValueError('Pinned parent protocol changed')
    for relative,expected in p['source_sha256'].items():
        if sha(PARENT/relative)!=expected:raise ValueError('Parent source changed')
    ck=Path(value['checkpoint'])
    if ck.parent.resolve()!=(PARENT_RUN/'checkpoints').resolve() or sha(ck)!=value['checkpoint_sha256']:
        raise ValueError('Parent checkpoint binding is corrupt')
    if value['format']!='adaptive-joint-next-event-v6':raise ValueError('Unsupported parent architecture')
    return value
