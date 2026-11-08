"""Seal stable deliverables last; raw sources remain bound by cohort manifests."""
from pathlib import Path
from .contracts import read,write,sha,now


def seal(root,study,extra=None):
    root=Path(root);art=root/'artifacts'
    if read(art/'status.json')['state']!='completed':raise ValueError('Do not seal unfinished study')
    if read(art/'power_guard.json')['state']!='released':raise ValueError('Wait for own sleep guard to release')
    files=set(root.glob('*.py'))|set(root.glob('*.md'))|set(art.glob('*.json'))
    files.discard(art/'delivery_manifest.json')
    for pattern in ('models/**/model.json','models/**/history.json','models/**/best.pt','models/**/*.txt',
                    'evaluation/**/*.json','logs/*.log','figures/*.png','figures/*.json'):
        files.update(art.glob(pattern))
    value=dict(created_at=now(),state='completed',study=study,freeze_sha256=sha(art/'freeze.json'),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,original_final_test_opened=False,
        files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)},**(extra or {}))
    target=art/'delivery_manifest.json'
    if target.exists():
        old=read(target)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Sealed delivery cannot change silently')
        return old
    write(target,value);print(study,'sealed',len(files),'files',sha(target));return value
