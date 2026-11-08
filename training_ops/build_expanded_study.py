"""Create a separate v7 study from the pinned v6 sources. Never edit v6."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'adaptive_joint'
NEW=ROOT/'adaptive_joint_expanded'


def replace(text, old, new):
    if text.count(old)!=1:raise ValueError('Source patch anchor differs: '+old[:90])
    return text.replace(old,new)


def main():
    protocol=json.loads((OLD/'artifacts/protocol.json').read_text(encoding='utf-8'))
    if (NEW/'artifacts/runs').exists():raise ValueError('Cannot rebuild an existing training study')
    NEW.mkdir(exist_ok=True)
    for relative,expected in protocol['source_sha256'].items():
        source=OLD/relative
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Parent source changed')
        target=NEW/relative;target.parent.mkdir(parents=True,exist_ok=True)
        text=source.read_text(encoding='utf-8')
        if relative.startswith('tests/'):
            text=text.replace('adaptive_joint','adaptive_joint_expanded')
            text=text.replace("self.assertEqual(set(map(len,folds.values())),{8})","self.assertEqual(sum(map(len,folds.values())),sum(m['joint_split']=='validation' for m in plan()['maps']))\n        self.assertLessEqual(max(map(len,folds.values()))-min(map(len,folds.values())),1)")
        text=text.replace('adaptive_v6_20261030','adaptive_v7_expanded100_20261031')
        if relative=='__init__.py':
            text=text.replace("VERSION = 'adaptive-joint-next-event-v6'","VERSION = 'adaptive-joint-next-event-v7-expanded100'")
        elif relative=='contracts.py':
            text=replace(text,"PLAN = LEGACY / 'configs/transformer_plan.json'","PLAN = ROOT.parent / 'additional_maps_20261030/expanded_plan.json'")
        elif relative=='world.py':
            for name in ('telemetry','contracts','baseline'):
                text=replace(text,'from adaptive_joint.'+name+' import','from .'+name+' import')
            text=replace(text,'from adaptive_joint import ARTIFACTS','from . import ARTIFACTS')
        elif relative=='model.py':
            text=replace(text,"        if warm_start:\n            ck = torch.load(PILOT, map_location='cpu', weights_only=False)\n            self.forecaster.load_state_dict(ck['model'], strict=True)\n",'')
            anchor="        self.selector_value = nn.Sequential(nn.Linear(PUBLIC_CONTEXT_DIM,32), nn.Tanh(), nn.Linear(32,1))"
            text=replace(text,anchor,anchor+'''
        if warm_start:
            from .initialization import parent_initialization
            initial = parent_initialization(required=True)
            ck = torch.load(initial['checkpoint'], map_location='cpu', weights_only=False)
            if ck.get('format') != initial['format'] or ck.get('purpose') != 'training' or ck['step'] != initial['step']:
                raise ValueError('Parent checkpoint identity differs from its frozen binding')
            if ck['protocol_sha256'] != initial['protocol_sha256']:
                raise ValueError('Parent checkpoint protocol differs')
            self.load_state_dict(ck['model'], strict=True)
''')
        elif relative=='experiment.py':
            text=replace(text,'from .model import Controller, optimizer','from .model import Controller, optimizer\nfrom .initialization import parent_initialization')
            text=replace(text,'protocol=dict(version=VERSION,runtime=runtime_contract(),','protocol=dict(version=VERSION,parent_initialization=parent_initialization(),runtime=runtime_contract(),')
            text=replace(text,"        warm_predictor=dict(path=str(PILOT),sha256=sha(PILOT),step=5500,\n                            scope='8-agent development data only; variable-fleet training required'),", "        warm_predictor=parent_initialization() or dict(status='pending_parent_completion'),\n        reference_pilot=dict(path=str(PILOT),sha256=sha(PILOT),step=5500,scope='Replay reference only'),")
            text=replace(text,'def assert_protocol(protocol,allow_final_test=False):','def assert_protocol(protocol,allow_final_test=False):\n    if protocol.get(\'parent_initialization\')!=parent_initialization():raise ValueError(\'Parent initialization binding changed; prepare again before training\')')
            text=text.replace("'adaptive-v6-train'","'adaptive-v7-expanded100-train'")
            start=text.index('    # Round-robin geographic sampling, not the first few scenarios of two maps.')
            end=text.index('\n\n\ndef prepare_assets',start)
            text=text[:start]+'''    # Mix original and added geography in every small validation batch.
    groups={}
    for s in specs:groups.setdefault(s['region'],[]).append(s)
    metadata={m['key']:m for m in plan()['maps']}
    original=sorted(k for k in groups if metadata[k].get('source_dataset','legacy_v6')=='legacy_v6')
    by_province={}
    for k in groups:
        if k not in original:by_province.setdefault(metadata[k].get('province','unknown'),[]).append(k)
    added=[]
    for j in range(max([len(v) for v in by_province.values()]+[0])):
        for province in sorted(by_province):
            keys=sorted(by_province[province])
            if j<len(keys):added.append(keys[j])
    order=[]
    for j in range(max(len(original),len(added))):
        if j<len(added):order.append(added[j])
        if j<len(original):order.append(original[j])
    ordered=[]
    for j in range(max(map(len,groups.values()))):
        for k in order:
            if j<len(groups[k]):
                rotation=order.index(k)%len(groups[k])
                ordered.append(groups[k][(j+rotation)%len(groups[k])])
    return ordered[:count]'''+text[end:]
        elif relative=='driver.py':
            anchor="    c=dict(protocol['config']);c['updates']=updates if updates is not None else c['updates']"
            text=replace(text,anchor,"    if not protocol.get('parent_initialization'):raise ValueError('Bind a completed parent study before training')\n"+anchor)
        compile(text,str(target),'exec')
        target.write_text(text,encoding='utf-8')
    c=dict(protocol['config']);c['seed']=2026103102
    (NEW/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(OLD/'requirements-lock.txt',NEW/'requirements-lock.txt')
    (NEW/'.gitignore').write_text('artifacts/\n__pycache__/\n*.pyc\n',encoding='utf-8')
    print(json.dumps(dict(study=str(NEW),files=len(protocol['source_sha256']),parent_source_pinned=True,training_started=False)))


if __name__=='__main__':main()
