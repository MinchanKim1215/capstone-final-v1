"""Two live packet views and actual parent composition, before V12 freeze."""
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,now,PARENT
from .wire import extend_trace
from .inference import parent_choices,parent_predict,predict
from . import runtime
from forecast_graph.contracts import source_records
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_graph.episodes import RawEpisode
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform


def main():
    #Training caches intentionally contain public-plan features only at the
    #predecessor's sampled training rows. They are not a full-flight oracle.
    #Recompute the causal public plan for this TRAIN-only streaming fixture.
    r=source_records()[0];ep=RawEpisode(r);n=ep.values.shape[2];ns=r['n_search'];city,nominal=public_context(r);city.radio.enable_cache()
    with np.load(PARENT/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
    with np.load(ARTIFACTS/'deltas'/(r['key']+'.npz'),allow_pickle=False) as a:newtx=extend_trace(tx,a['delta'])
    received=np.unpackbits(rx,axis=-1)[...,:n];ends=np.arange(40,121,2);lookup={int(e):t for t,e in enumerate(ep.ends)}
    local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);views={}
    for profile,packets in [('original',tx),('mission',newtx)]:
        a=build(local,ReceivedTrace(packets,rx,n,ns),np.c_[ends,np.zeros(len(ends),int)],city)
        a.update(n=np.full(len(ends),n),ns=np.full(len(ends),ns),observer=np.full(len(ends),ns),query=np.c_[[lookup[int(e)] for e in ends],np.zeros(len(ends),int)])
        views[profile]=a
    model=str(PARENT/'models/v11_hybrid_shared/model.json');new=predict(model,views['mission'],device='cuda')
    choices=parent_choices();tasks={};expected={};policy=ARTIFACTS/'preflight_identity_policy.json'
    write(policy,dict(parameters={'2':dict(slope=1.,intercept=0.),'5':dict(slope=1.,intercept=0.)},threshold=.5,require_energy=True))
    for task in ('link','service'):
        parent=choices[task]['champion']['candidate'];old=parent_predict(parent,views['original'],r,ep)
        value=new[task][:,:n] if task=='link' else new[task]
        expected[task]=transform((old+value)*.5,read(policy)['parameters'])
        a=dict(task=task,tag='prototype_parent',family='parent',parent_candidate=parent)
        b=dict(task=task,tag='prototype_mission',family='new',profile='mission',model=model)
        tasks[task]=dict(candidate=dict(task=task,tag='prototype_blend',family='blend',members=[a,b]),policy=str(policy))
    runtime.bundle=lambda:dict(tasks=tasks)
    streams={task:runtime.Stream(city,n,ns,task,nominal) for task in tasks}
    maximum={task:0. for task in tasks};features=0.;tested=0;feature_detail={profile:{k:0. for k in ('node_shared','edge_shared','physics','global')} for profile in views};old_detail={};old_columns={}
    for q in range(120):
        packets=[newtx[q//2,j] for j in range(n) if j!=ns and received[q//2,ns,j]] if q%2==0 else []
        own=newtx[q//2,ns] if q%2==0 else None
        for task,stream in streams.items():
            out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own)
            if out['status']!='ready':continue
            t=(q+1-40)//2
            if task=='link':
                peers=[j for j in range(n) if j!=ns];batchrows=np.array([[lookup[q+1],0,j] for j in peers]);liverows=np.array([[0,0,j] for j in peers])
                aa=ep.features(batchrows,True,True);bb=stream.local.features(liverows,True,True)
                for col in np.flatnonzero(np.any(~np.isclose(aa,bb,rtol=0,atol=0,equal_nan=True),axis=0)):
                    difference=np.abs(aa[:,col]-bb[:,col]);finite=difference[np.isfinite(difference)]
                    old_columns[str(col)]=max(old_columns.get(str(col),0.),float(finite.max()) if len(finite) else 0.)
                seq1=ep.sequence(batchrows,True);seq2=stream.local.sequence(liverows,True)
                old_detail['sequence_max']=max(old_detail.get('sequence_max',0.),float(np.max(np.abs(seq1-seq2))))
            for profile in views:
                for key in ('node_shared','edge_shared','physics','global'):
                    difference=float(np.abs(stream.last_inputs[profile][key].astype(float)-views[profile][key][t:t+1].astype(float)).max())
                    features=max(features,difference);feature_detail[profile][key]=max(feature_detail[profile][key],difference)
            if task=='link':
                for row in out['links']:
                    for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                        if row[field] is not None:maximum[task]=max(maximum[task],abs(row[field]-float(expected[task][t,row['peer'],h])));tested+=1
            else:
                for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                    if out[field] is not None:maximum[task]=max(maximum[task],abs(out[field]-float(expected[task][t,h])));tested+=1
    diagnostic={'feature_difference':feature_detail,'probability_difference':maximum,'old_static_columns':old_columns,'old_detail':old_detail}
    write(ARTIFACTS/'stream_preflight_diagnostic.json',diagnostic);print(diagnostic,flush=True)
    assert features==0 and all(x<1e-4 for x in maximum.values())
    value=dict(passed=True,prototype_only=True,frames=120,maximum_feature_difference=features,maximum_probability_difference=maximum,tested_predictions=tested,created_at=now())
    write(ARTIFACTS/'stream_preflight.json',value);print(value)


if __name__=='__main__':main()
