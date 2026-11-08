"""Past-only inputs, unsent information invariance and actual receiver replay."""
from types import SimpleNamespace
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,WORKSPACE
from .contracts import read,write,sha,now,milestone
from .state import from_replay
from .wire import Received,SLOTS
from .features import summarize,one as network_one,tree_features
from .physics import Geometry
from forecast_graph.episodes import physical
from forecast_graph.features import LocalHistory,ReceivedTrace
from forecast_service.features import build as tree_base
from forecast_rebuild.planning import public_context

def main(require_full=True):
    parents=read(PARENT/'data_manifest.json')['records'];parent=next(p for p in parents if p['role']=='selection');item=read(ARTIFACTS/'data'/(parent['key']+'.json'));r=item['source_record'];ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];city,_=public_context(r);city.radio.enable_cache()
    with np.load(item['path']) as a:ends=a['ends'];queries=a['query'];cached={k:a[k] for k in ('base','network','physics','network_history')}
    replay=read(item['replay_path'][:-4]+'.json')
    with np.load(replay['path']) as a:tx=a['tx'];rx=a['rx'];own=a['observed']
    network,known=from_replay(replay['path']);summary=summarize(network,known,n,ns);np.testing.assert_equal(summary.astype(np.float16),cached['network_history']);np.testing.assert_equal(tree_features(summary,ends),cached['network'])
    # Independent incremental receiver, exact existing beacon delivery clock.
    cache=Received(n,ns);decoded=np.unpackbits(rx,axis=-1)[...,:n];incremental=[]
    for q in range(min(120,len(own))):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and decoded[q//2,ns,j]] if q%2==0 else [];own_packet=tx[q//2,ns] if q%2==0 else None;cache.observe((q+1)*.5,packets,own_packet);v,m=cache.snapshot((q+1)*.5,own[q,ns]);incremental.append(network_one(v,m,n,ns))
    np.testing.assert_equal(np.stack(incremental),summary[:len(incremental)])
    positions=np.unique(np.linspace(0,len(ends)-1,3,dtype=int));chosen=ends[positions];old_tx=tx.copy();old_tx[...,SLOTS]=0;old_tx[...,127]=0;trace=ReceivedTrace(old_tx,rx,n,ns);local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns)
    calculated=Geometry(city).build(local,trace,chosen);np.testing.assert_equal(calculated,cached['physics'][positions])
    q=0;end=int(chosen[q]);query=int(queries[positions[q]]);original=calculated[q]
    values=np.array(ep.values,copy=True);masks=np.array(ep.masks,copy=True);intents=np.array(ep.intents,copy=True);energy=np.array(ep.energy,copy=True)
    values[~masks]+=73;values[end:]+=73;intents[end:]+=99;energy[end:]-=99
    altered=old_tx.copy();latest=(end-1)//2;altered[latest+1:,:,30:114]^=np.uint8(3)
    not_received=~decoded[:,ns].astype(bool);not_received[:,ns]=False
    for t,j in np.argwhere(not_received):altered[t,j,30:114]^=np.uint8(7)
    changed_local=LocalHistory(values,masks,intents,n,ns);changed_trace=ReceivedTrace(altered,rx,n,ns)
    np.testing.assert_equal(Geometry(city).one(changed_local,changed_trace,end),original)
    changed_ep=SimpleNamespace(values=values,masks=masks,intents=intents,energy=energy,ends=ep.ends)
    original_base=tree_base(ep,[[query,0]],ns,city);altered_base=tree_base(changed_ep,[[query,0]],ns,city);np.testing.assert_equal(original_base,altered_base)
    # Prefix-only serialization is already independently checked against a
    # truncated physical replay in instrumentation_preflight.json.
    all_records=read(ARTIFACTS/'data_manifest.json')['records'] if (ARTIFACTS/'data_manifest.json').exists() else []
    checked=0
    if require_full:
        if len(all_records)!=335:raise ValueError('Stage2requires all335development records')
        for x in all_records:
            if sha(x['path'])!=x['sha256']:raise ValueError('Inputcache changed')
            with np.load(x['path']) as a,np.load(x['parent_path']) as p:
                for k in ('ends','query','y','up','support','links','uniform','bags'):np.testing.assert_equal(a[k],p[k])
                if a['base'].shape!=(x['queries'],434) or a['network'].shape!=(x['queries'],1116) or a['physics'].shape!=(x['queries'],54):raise ValueError('Inputdimension mismatch')
            checked+=1
    value=dict(passed=True,created_at=now(),case=r['key'],incremental_receiver_frames=len(incremental),incremental_network_summary_exact=True,
        cached_physics_queries_exact=len(positions),future_RF_intent_energy_and_unreceived_beacon_perturbations_exact=True,unobserved_RF_perturbation_exact=True,
        full335cohort_checked=require_full,records_checked=checked,labels_and_event_denominators_unchanged=True,
        source_pins={k:sha(ROOT/k) for k in ('features.py','state.py','physics.py','wire.py','data.py')})
    write(ARTIFACTS/('input_preflight.json' if require_full else 'input_fixture_checks.json'),value)
    if require_full:milestone(2,'335개 입력·정답 일치, 실제 수신 순서·관측 불가/미래값 교란 불변: artifacts/input_preflight.json','동일 입력 대조군·새 큐 관측·경로 불확실성의 효과를 분리해 학습하고 선택45·보정45에서 후보와 문턱을 고정한다.')
    print(value,flush=True)

if __name__=='__main__':main()
