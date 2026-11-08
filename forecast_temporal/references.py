"""Read-only pure frozen V12 service baseline on genuinely new flights."""
import numpy as np
import torch
from . import ARTIFACTS,NAV
PARENT=NAV
from .contracts import read,write,sha,now
from forecast_recall.inference import parent_entry
from forecast_graph.episodes import RawEpisode
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_graph.inference import predict
from forecast_graph.contracts import V10
from forecast_service.features import build as tree_features
from forecast_service.inference import tree_predict


def navigation_reference(item):
    key=item['key'];folder=ARTIFACTS/'navigation_references';folder.mkdir(exist_ok=True);path=folder/(key+'.npz');meta=path.with_suffix('.json')
    contract=dict(data_sha256=item['sha256'],parent_freeze_sha256=sha(PARENT/'freeze.json'),implementation_sha256=sha(__file__))
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Parent reference changed')
        return dict(item,reference_path=str(path),reference_sha256=old['sha256'])
    r=item['source_record'];ep=RawEpisode(r);n=ep.values.shape[2];ns=r['n_search'];ep.city.radio.enable_cache()
    with np.load(item['path'],allow_pickle=False) as a:q=a['query'];ends=a['ends'];mission=torch.sigmoid(torch.from_numpy(a['logits'])).numpy().mean(1)
    with np.load(ARTIFACTS/'fresh_traces'/(key+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
    local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);trace=ReceivedTrace(tx,rx,n,ns)
    pieces={};parts={k:[] for k in ('node_shared','edge_shared','physics','global')}
    for start in range(0,len(q),128):
        e=ends[start:start+128];a=build(local,trace,np.c_[e,np.zeros(len(e),int)],ep.city)
        for k in parts:parts[k].append(a[k])
    arrays={k:np.concatenate(v) for k,v in parts.items()};arrays.update(n=np.full(len(q),n),ns=np.full(len(q),ns),observer=np.full(len(q),ns))
    def v11(candidate):
        if candidate['family']=='blend':return sum(v11(c)*.5 for c in candidate['members'])
        if candidate['family']=='previous':
            model=read(V10/'research_candidates.json')['tasks']['service']['model'];x=tree_features(ep,np.c_[q,np.zeros(len(q),int)],ns,ep.city);return tree_predict(model,x)
        return predict(candidate['model'],arrays,device='cpu')['service']
    def v12(candidate):
        if candidate['family']=='blend':return sum(v12(c)*.5 for c in candidate['members'])
        if candidate['family']=='parent':return v11(candidate['parent_candidate'])
        if candidate['model']!=str(PARENT/'models/v12_mission/model.json'):raise ValueError('Unexpected baseline model')
        return mission
    prediction=v12(parent_entry()['candidate'])[:,None,None];np.savez_compressed(path,prediction=prediction)
    write(meta,dict(contract=contract,sha256=sha(path),created_at=now()))
    return dict(item,reference_path=str(path),reference_sha256=sha(path))


def prepare_case(record):
    from .legacy_data import prepare_one as legacy_prepare
    from .data import prepare_one
    from .inference import parent_entry as v13_entry
    from forecast_recall.inference import predict as old_predict
    legacy=legacy_prepare(dict(key=record['key'],role='confirmation',region=record['region'],source_record=record))
    legacy=navigation_reference(legacy)
    with np.load(legacy['path']) as a:inputs={k:a[k] for k in ('embedding','logits','history','ends')}
    candidate=v13_entry()['candidate']
    if candidate['family']!='new':raise ValueError('Expected fixed V13 memory model')
    prediction=old_predict(candidate['model'],inputs,device='cpu')[:,None,None]
    path=ARTIFACTS/'references'/(record['key']+'.npz');path.parent.mkdir(exist_ok=True)
    if not path.exists():np.savez_compressed(path,prediction=prediction)
    else:
        with np.load(path) as a:np.testing.assert_equal(a['prediction'],prediction)
    item=prepare_one(legacy)
    return dict(item,legacy_path=legacy['path'],legacy_sha256=legacy['sha256'],navigation_reference_path=legacy['reference_path'],navigation_reference_sha256=legacy['reference_sha256'],reference_path=str(path),reference_sha256=sha(path))
