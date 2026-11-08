"""실제 가중치로 배치/로컬/지도 내부 계산의 결과 일치 및 CPU 지연 측정."""
import time,json
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json
from usable.data import collection,load_observation
from usable.predictor import LinkRiskPredictor
from usable.evaluate import RESULT,RUNS


def main():
    selected=read_json(RESULT/'selection.json')['selected_variant']+'_s11';folder=RUNS/selected
    manifest,p=collection();r=next(r for r in manifest['records'] if r['region']=='seoul_20261023_03' and r['metadata']['direction']=='north_south' and r['metadata']['kind']=='intermittent')
    entry=next(e for e in p['maps'] if e['key']==r['region']);x,m=load_observation(r)
    with np.load(ROOT/r['path']/'spatial_observations.npz') as z:xx=np.concatenate((x,z['features']),-1);mm=np.concatenate((m,z['mask']),-1)
    a=LinkRiskPredictor(folder/'best.pt',threads=1)
    b=LinkRiskPredictor(folder/'best.pt',threads=1,map_path=ROOT/entry['path'])
    c=LinkRiskPredictor(folder/'best.pt',threads=1)
    diff=0.;localdiff=0.;elapsed=[];map_elapsed=[];local_elapsed=[];decisions=0
    for t in range(340):
        now=(t+1)*.5
        before=time.perf_counter();ra=a.observe(now,xx[t],mm[t]);ta=time.perf_counter()-before
        before=time.perf_counter();rb=b.observe(now,x[t],m[t]);tb=time.perf_counter()-before
        before=time.perf_counter();rc=c.observe_local(now,0,xx[t,0],mm[t,0]);tc=time.perf_counter()-before
        if ra['status']!='ready':continue
        elapsed.append(ta);map_elapsed.append(tb);local_elapsed.append(tc);decisions+=1
        for u,v in zip(ra['links'],rb['links']):
            assert u['status']==v['status']
            if u['risk_5s'] is not None:
                diff=max(diff,abs(u['risk_5s']-v['risk_5s']))
        own=[u for u in ra['links'] if u['source']==0]
        for u,v in zip(own,rc['links']):
            assert u['status']==v['status']
            if u['risk_5s'] is not None:localdiff=max(localdiff,abs(u['risk_5s']-v['risk_5s']))
    assert diff<2e-5 and localdiff<2e-5
    def latency(a):return {'median_ms':float(np.median(a)*1000),'p95_ms':float(np.percentile(a,95)*1000)}
    out={'run':selected,'episode':r['id'],'checked_decisions':decisions,
         'precomputed_vs_internal_map_max_probability_difference':diff,
         'full_batch_vs_local_only_max_probability_difference':localdiff,
         'status_matches':True,'cpu_threads':1,'all_56_links':latency(elapsed),
         'all_56_links_with_map_computation':latency(map_elapsed),'one_observer_7_links':latency(local_elapsed),
         'scope':'this CPU process, no real flight controller or radio hardware timing claim'}
    write_json(RESULT/'runtime_verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
