"""A completed sealed run is read-only when a campaign is accidentally resumed."""
from . import ROOT,ARTIFACTS
from .contracts import read,sha

def already_sealed():
    path=ARTIFACTS/'delivery_manifest.json'
    if not path.exists():return False
    manifest=read(path)
    for relative,item in manifest['files'].items():
        if sha(ROOT/relative)!=item['sha256']:raise ValueError('Sealed V16 delivery changed: '+relative)
    for name,item in manifest.get('supplementary_outputs',{}).items():
        if sha(name)!=item['sha256']:raise ValueError('Sealed supporting output changed: '+name)
    print('V16 already completed and sealed; verified unchanged. Report:',ROOT/'RESULTS_20261103.md',flush=True)
    return True
