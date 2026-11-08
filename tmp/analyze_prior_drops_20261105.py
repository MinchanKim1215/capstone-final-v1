"""Full-flight, past-only precursor diagnosis with normal-time controls."""
from pathlib import Path
import sys
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_joint_expanded import LEGACY,ARTIFACTS as NAV
from adaptive_joint_expanded.baseline import paired_shadow
from adaptive_joint_expanded.features import THRESHOLD,widest
from forecast_shield.forecaster import geometry
from forecast_avoidance.controller import fragile_margin
from forecast_prevention.diagnostics import connected,outage_runs
from forecast_rebuild.contracts import read,write,sha,now,maps
from usable.collect import prepare_city_without_unused_radio_cache
from swarm_sim.config import Scenario
from outage_atlas.radio import probabilities,radio_graph

OUT=ROOT/'output/precursor_drops_20261105'
ATLAS=ROOT/'outage_atlas/artifacts'
CAUSES=ROOT/'final_v1/artifacts/causes.json'
RULES=dict(sample_s=.5,drop_db=6.,history_s=30.,minimum_lead_s=3.,future_s=30.,
           burst_quiet_s=3.,recovery_within_db=3.,recovery_duration_s=2.,
           post_margin_bands_db=[0.,3.,10.,20.],
           query_scope='Current graph connected, 30 s complete past and 30 s complete future plus persistence check',
           clean_query='Every actual network frame in the preceding 30 s is connected',
           purpose='Explanatory upper-bound diagnosis from simulated whole-network RF truth; '
                   'not receiver inputs, learned-model validation, or a relocation experiment')


def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=scalar),encoding='utf-8')


def rolling_sum(flag,length):
    cumulative=np.r_[0,np.cumsum(flag,dtype=np.int64)]
    left=np.maximum(0,np.arange(len(flag))+1-length)
    return cumulative[1:]-cumulative[left]


def rolling_peak(values,length):
    # Values before flight start are unavailable, never made into observations.
    padded=np.r_[np.full(length-1,-np.inf),values]
    return np.lib.stride_tricks.sliding_window_view(padded,length).max(axis=1)


def pulse_mask(margin,network,threshold=6.):
    drop=np.r_[0.,margin[:-1]-margin[1:]]
    previous=np.r_[False,network[:-1]]
    return (drop>=threshold)&network&previous&(margin>=0.)


def full_trace(entry,registry):
    if sha(entry['path'])!=entry['sha256']:
        raise ValueError('Source metadata changed')
    meta=read(entry['path']);array=meta['archive_path']
    if sha(array)!=meta['archive_sha256']:
        raise ValueError('Flight archive changed')
    path=LEGACY/registry['path']
    if sha(path)!=registry['sha256']:
        raise ValueError('Map changed')
    scenario=Scenario(direction=meta['spec']['direction'],motion_dt_s=3).validate()
    city=prepare_city_without_unused_radio_cache(path,scenario,NAV/'cache/city')
    city.radio.enable_cache()
    with np.load(array,allow_pickle=False) as z:
        nodes=z['nodes'];links=z['actual_links'];nt=len(links)
        service=np.unpackbits(z['service_'+meta['spec']['kind']])[:nt].astype(bool)
    frames=np.arange(nt)
    step,fraction=np.divmod(frames,6)
    fraction=fraction[:,None,None]/6.
    positions=city.points[nodes]
    xy=positions[step]*(1.-fraction)+positions[step+1]*fraction
    loss,extra=geometry(city,scenario,xy)
    n=len(nodes[0])
    shadow=paired_shadow(meta['spec']['seed'],nt,n,.5,scenario.shadow_sigma_db,scenario.shadow_correlation_s)
    noise=np.zeros((nt,n))
    jammer=meta['jammer']
    if meta['spec']['kind']!='clear':
        receivers=np.c_[xy.reshape(-1,2),np.full(nt*n,50.)]
        jl=city.radio.propagation(np.array([jammer['jammer_xy_m']+[2.]]),receivers,scenario)[0]
        noise=10**((jammer['tx_power_dbm']-jl.reshape(nt,n))/10.)
        if meta['spec']['kind']=='intermittent':
            active=(frames*.5>=jammer['intermittent_start_s'])&(frames*.5<jammer['intermittent_stop_s'])
            noise*=active[:,None]
    rebuilt=radio_graph(probabilities(loss,noise,shadow))
    np.testing.assert_array_equal(rebuilt,links)
    noise_db=10*np.log10(10**(-10.7)+noise)
    directed=10.-loss-shadow-noise_db[:,None,:]-THRESHOLD
    graph=np.minimum(directed,directed.swapaxes(-1,-2))
    margin=widest(graph,meta['mission']['search_drones']).min(axis=-1)
    network=connected(links,meta['mission']['search_drones'])
    return meta,margin,network,service,directed,graph,loss,extra,shadow,noise_db,links


def process(task):
    entry,registry,old=task
    destination=OUT/'maps'/(entry['region']+'.json')
    if destination.exists():
        saved=read(destination)
        if saved['contract']!=dict(metadata_sha256=entry['sha256'],script_sha256=sha(__file__),rules=RULES):
            raise ValueError('Cached analysis contract changed')
        if sha(saved['trace_path'])!=saved['trace_sha256']:
            raise ValueError('Cached diagnostic trace changed')
        return saved
    meta,margin,network,service,directed,graph,loss,extra,shadow,noise_db,links=full_trace(entry,registry)
    nt=len(margin);ns=meta['mission']['search_drones']
    events=outage_runs(network)
    if [(a*.5,b*.5) for a,b in events]!=[(e['onset_s'],e['end_s']) for e in old['events']]:
        raise ValueError('Persistent events changed')
    for (a,b),prior in zip(events,old['events']):
        np.testing.assert_array_equal(np.round(margin[max(0,a-12):a+1],2),prior['margin_db'])
    raw=np.r_[0.,margin[:-1]-margin[1:]]
    pulse=pulse_mask(margin,network,RULES['drop_db'])
    # Causal grouping: count a new drop only after 3 s without a qualified drop.
    prior_pulse=rolling_sum(pulse,6)-pulse
    starts=pulse & (prior_pulse==0)
    pidx=np.flatnonzero(starts)
    full=(pidx+62<nt)
    chosen=pidx[full]
    starts_array=np.array([a for a,b in events if a>0],int)
    next_index=np.searchsorted(starts_array,np.arange(nt),side='right')
    next_start=np.full(nt,np.inf)
    has=next_index<len(starts_array)
    next_start[has]=starts_array[next_index[has]]
    lead=(next_start-np.arange(nt))*.5
    last_down=np.maximum.accumulate(np.where(~network,np.arange(nt),-1))
    precursor=[]
    for (a,b),prior in zip(events,old['events']):
        if a==0:continue
        left=max(1,a-60);right=a-6
        window=np.arange(left,right+1) if right>=left else np.array([],int)
        any_drop=window[raw[window]>=6.]
        up_drop=window[pulse[window]]
        clean_drop=up_drop[up_drop>last_down[a-1]]
        precursor.append(dict(pattern=prior['pattern'],onset_s=a*.5,full30=a>=60,
             any_prior_jump=bool(len(any_drop)),still_connected_jump=bool(len(up_drop)),
             uninterrupted_to_onset_jump=bool(len(clean_drop)),
             most_recent_clean_lead_s=(a-int(clean_drop[-1]))*.5 if len(clean_drop) else None,
             prior_persistent_outage=any(y>max(0,a-60) for x,y in events if y<=a)))
    backup=fragile_margin(graph[chosen],ns) if len(chosen) else np.zeros(0)
    binary_backup=fragile_margin(np.where(links[chosen],1.,-1.),ns)>=0 if len(chosen) else np.zeros(0,bool)
    counterfactual={}
    for term,delta in (
        ('building',extra[chosen]-extra[chosen-1]),
        ('distance',(loss[chosen]-extra[chosen])-(loss[chosen-1]-extra[chosen-1])),
        ('shadow',shadow[chosen]-shadow[chosen-1]),
        ('interference',(noise_db[chosen]-noise_db[chosen-1])[:,None,:])):
        alternative=directed[chosen]+delta
        value=widest(np.minimum(alternative,alternative.swapaxes(-1,-2)),ns).min(axis=-1)
        counterfactual[term]=value-margin[chosen]
    pulses=[]
    ii,jj=np.triu_indices(links.shape[-1],1)
    for k,t in enumerate(chosen):
        recovery=None
        level=margin[t-1]-RULES['recovery_within_db']
        condition=(margin[t:t+64]>=level)&network[t:t+64]
        stable=np.flatnonzero(np.convolve(condition.astype(int),np.ones(4,dtype=int),'valid')==4)
        if len(stable) and stable[0]*.5<=30:
            recovery=float(stable[0]*.5)
        delay=float(lead[t]) if np.isfinite(lead[t]) and lead[t]<=30 else None
        outcome='late_under3' if delay is not None and delay<3 else 'early_3to7' if delay is not None and delay<7 else 'long_7to30' if delay is not None else 'no_onset_30'
        pulses.append(dict(region=entry['region'],kind=meta['spec']['kind'],time_s=t*.5,
            drop_db=float(raw[t]),before_db=float(margin[t-1]),after_db=float(margin[t]),
            first_onset_lead_s=delay,outcome=outcome,recovery_start_s=recovery,
            previous30_all_connected=bool(network[max(0,t-60):t+1].all() and t>=60),
            prior_link_losses=int((links[t-1,ii,jj]&~links[t,ii,jj]).sum()),
            single_link_backup=bool(binary_backup[k]),backup_margin_db=float(backup[k]),
            component_restoration_db={name:float(value[k]) for name,value in counterfactual.items()},
            service_bad_next30=bool((~service[t+1:t+61]).any()),
            service_bad_now=bool(not service[t])))
    recent=rolling_sum(pulse,61)>0
    peak=rolling_peak(margin,61)
    unrestored=recent&(peak-margin>=6.)
    valid=(np.arange(nt)>=60)&(np.arange(nt)+62<nt)&network&(margin>=0.)
    clean=valid&(rolling_sum(~network,61)==0)
    band=np.digitize(margin,[3.,10.,20.])
    features={'recent_drop30':recent,'recent_drop_unrestored':unrestored,
              'low3':margin<3.,'drop_and_low3':recent&(margin<3.),
              'unrestored_and_low3':unrestored&(margin<3.)}
    queries={}
    for scope,mask in (('all_currently_up',valid),('continuous_up30',clean)):
        scopes={}
        for name,feature in features.items():
            bins=[]
            for b in range(4):
                selected=mask&(band==b)
                positive=selected&feature;negative=selected&~feature
                bins.append(dict(n1=int(positive.sum()),y1=int((positive&(lead>0)&(lead<=30)).sum()),
                     early1=int((positive&(lead>=3)&(lead<=30)).sum()),
                     n0=int(negative.sum()),y0=int((negative&(lead>0)&(lead<=30)).sum()),
                     early0=int((negative&(lead>=3)&(lead<=30)).sum())))
            scopes[name]=bins
        queries[scope]=dict(eligible_frames=int(mask.sum()),features=scopes)
    trace=OUT/'traces'/(entry['region']+'.npz');trace.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trace,margin_db=margin.astype(np.float32),network=network,service=service,
                        pulse=pulse,pulse_group_start=starts)
    result=dict(region=entry['region'],kind=meta['spec']['kind'],split=entry['split'],
        spatial_group=registry.get('joint_spatial_group',entry['region']),frames=nt,duration_s=nt*.5,
        new_outages=len(starts_array),initial_outages=sum(a==0 for a,b in events),
        raw_drops=int((raw>=6.).sum()),connected_raw_drops=int(pulse.sum()),
        grouped_full_future_pulses=len(chosen),censored_grouped_pulses=int((~full).sum()),
        pulses=pulses,precursors=precursor,queries=queries,
        threshold_sensitivity={str(d):int(pulse_mask(margin,network,d).sum()) for d in (4.,6.,8.)},
        trace_path=str(trace),trace_sha256=sha(trace),
        contract=dict(metadata_sha256=entry['sha256'],script_sha256=sha(__file__),rules=RULES),
        exact_original_graph=True,exact_original_six_second_trace=True)
    save(destination,result)
    return result


def aggregate(results):
    pulses=[p for r in results for p in r['pulses']]
    precursors=[p for r in results for p in r['precursors']]
    shapes={}
    for shape in ('abrupt','gradual','mixed','other','insufficient_history'):
        rows=[p for p in precursors if p['pattern']==shape]
        complete=[p for p in rows if p['full30']]
        shapes[shape]=dict(n=len(rows),n30=len(complete),
            all_events={key:sum(p[key] for p in rows) for key in ('any_prior_jump','still_connected_jump','uninterrupted_to_onset_jump','prior_persistent_outage')},
            complete30={key:sum(p[key] for p in complete) for key in ('any_prior_jump','still_connected_jump','uninterrupted_to_onset_jump','prior_persistent_outage')})
    groups={}
    for clean in (False,True):
        selected=[p for p in pulses if not clean or p['previous30_all_connected']]
        rows={}
        for b,label in enumerate(('0to3','3to10','10to20','20plus')):
            subset=[p for p in selected if (p['after_db']<3 if b==0 else 3<=p['after_db']<10 if b==1 else 10<=p['after_db']<20 if b==2 else p['after_db']>=20)]
            n=len(subset);y=sum(p['first_onset_lead_s'] is not None for p in subset)
            rows[label]=dict(n=n,onset_next30=y,risk=y/n if n else None,
                onsets_3to30=sum(p['first_onset_lead_s'] is not None and p['first_onset_lead_s']>=3 for p in subset),
                outcomes=dict(Counter(p['outcome'] for p in subset)),
                recovered=sum(p['recovery_start_s'] is not None for p in subset),
                backup_counts={str(v):dict(n=sum(p['single_link_backup']==v for p in subset),
                     next30=sum(p['single_link_backup']==v and p['first_onset_lead_s'] is not None for p in subset)) for v in (False,True)})
        groups['no_prior_down30' if clean else 'all']=dict(n=len(selected),bands=rows,
            outcomes=dict(Counter(p['outcome'] for p in selected)),
            recovered_within30=sum(p['recovery_start_s'] is not None for p in selected),
            no_onset_recovered=sum(p['outcome']=='no_onset_30' and p['recovery_start_s'] is not None for p in selected),
            no_onset_unrecovered=sum(p['outcome']=='no_onset_30' and p['recovery_start_s'] is None for p in selected),
            link_lost_but_network_up=sum(p['prior_link_losses']>0 for p in selected),
            component_explains_half={k:sum(p['component_restoration_db'][k]>=.5*p['drop_db'] for p in selected) for k in ('building','distance','shadow','interference')})
    controls={}
    for scope in ('all_currently_up','continuous_up30'):
        features={}
        for name in results[0]['queries'][scope]['features']:
            bands=[]
            for b in range(4):
                tables=[r['queries'][scope]['features'][name][b] for r in results]
                pooled={key:sum(t[key] for t in tables) for key in tables[0]}
                pooled.update(risk1=pooled['y1']/pooled['n1'] if pooled['n1'] else None,
                              risk0=pooled['y0']/pooled['n0'] if pooled['n0'] else None)
                # Compare within each map and current-margin band. Shared min(n1,n0)
                # weights prevent different map/region mixtures from driving the result.
                common=[(t,min(t['n1'],t['n0'])) for t in tables if t['n1']>=20 and t['n0']>=20]
                total=sum(w for t,w in common)
                r1=sum(t['y1']/t['n1']*w for t,w in common)/total if total else None
                r0=sum(t['y0']/t['n0']*w for t,w in common)/total if total else None
                pooled.update(common_maps=len(common),standardized_risk1=r1,standardized_risk0=r0,
                              standardized_risk_ratio=r1/r0 if r0 else None)
                bands.append(pooled)
            features[name]=bands
        controls[scope]=dict(frames=sum(r['queries'][scope]['eligible_frames'] for r in results),features=features)
    summary=dict(created_at=now(),rules=RULES,maps=len(results),radio_frames=sum(r['frames'] for r in results),
        hours=sum(r['duration_s'] for r in results)/3600.,new_outages=len(precursors),
        raw_drops=sum(r['raw_drops'] for r in results),connected_raw_drops=sum(r['connected_raw_drops'] for r in results),
        grouped_full_future_pulses=len(pulses),censored_grouped_pulses=sum(r['censored_grouped_pulses'] for r in results),
        types=shapes,pulse_groups=groups,controls=controls,
        verified=dict(all_original_graphs=all(r['exact_original_graph'] for r in results),
                      all_original_6s_values=all(r['exact_original_six_second_trace'] for r in results)),
        manifest_sha256=sha(ATLAS/'manifest.json'),causes_sha256=sha(CAUSES),script_sha256=sha(__file__))
    save(OUT/'summary.json',summary)
    save(OUT/'pulse_records.json',pulses)
    save(OUT/'precursor_records.json',precursors)
    save(OUT/'completed.json',dict(state='completed',created_at=now(),maps=len(results),
         summary_sha256=sha(OUT/'summary.json'),radio_frames=summary['radio_frames']))
    print(json.dumps({k:summary[k] for k in ('maps','radio_frames','hours','new_outages','raw_drops','connected_raw_drops','grouped_full_future_pulses','types','pulse_groups')},ensure_ascii=False,indent=2,default=scalar),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_at=now(),rules=RULES,source_manifest_sha256=sha(ATLAS/'manifest.json'),
        source_causes_sha256=sha(CAUSES),script_sha256=sha(__file__),
        unchanged_operational_controller=True,new_training=False,
        interpretation='All 345 maps are diagnostic-only for this task. No fitting or threshold selection. '
                       'Conditional associations under the already active V7 policy are not causal prevention effects.')
    p=OUT/'protocol.json'
    if p.exists():
        old=read(p)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in protocol.items() if k!='created_at'}:
            raise ValueError('Analysis contract changed; preserve old output and start a new version')
    else:save(p,protocol)
    causes=read(CAUSES);prior={r['region']:r for r in causes['flights']};registry=maps()
    entries=read(ATLAS/'manifest.json')['records']
    if len(entries)!=345:raise ValueError('Unexpected atlas coverage')
    results={}
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(process,(e,registry[e['region']],prior[e['region']])):i for i,e in enumerate(entries)}
        for count,job in enumerate(as_completed(jobs),1):
            results[jobs[job]]=job.result()
            if count%12==0 or count==len(entries):
                save(OUT/'status.json',dict(state='running',complete=count,total=len(entries),updated_at=now()))
                print('full-flight precursor maps',count,'/',len(entries),flush=True)
    aggregate([results[i] for i in sorted(results)])
    save(OUT/'status.json',dict(state='completed',complete=345,total=345,updated_at=now()))


if __name__=='__main__':main()
