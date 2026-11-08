"""Same-map, 1 dB current-level sensitivity analysis; no fitting."""
from pathlib import Path
import sys
from collections import defaultdict,Counter
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from forecast_rebuild.contracts import read,write,sha,now,maps
from forecast_prevention.diagnostics import outage_runs
from forecast_avoidance.controller import fragile_margin
from importlib.util import spec_from_file_location,module_from_spec

spec=spec_from_file_location('precursor_core',ROOT/'tmp/analyze_prior_drops_20261105.py')
core=module_from_spec(spec);spec.loader.exec_module(core)
OUT=core.OUT


def standardized(tables,seed):
    grouped=defaultdict(lambda:np.zeros(5))
    strata=0
    for group,table in tables:
        n1,y1,n0,y0,e1,e0=table
        if n1<20 or n0<20:continue
        w=min(n1,n0);strata+=1
        grouped[group]+=np.array([w,w*y1/n1,w*y0/n0,w*e1/n1,w*e0/n0])
    if not grouped:return dict(groups=0,strata=0)
    a=np.array(list(grouped.values()))
    pooled=a.sum(axis=0)
    rng=np.random.default_rng(seed)
    boot=a[rng.integers(len(a),size=(5000,len(a)))].sum(axis=1)
    output=dict(groups=len(a),strata=strata,common_weight=float(pooled[0]))
    for prefix,i,j in (('next30',1,2),('early3to30',3,4)):
        ratio=pooled[i]/pooled[j] if pooled[j] else None
        valid=boot[:,j]>0
        rr=boot[valid,i]/boot[valid,j]
        difference=(boot[:,i]-boot[:,j])/boot[:,0]
        output[prefix]=dict(risk_with_history=float(pooled[i]/pooled[0]),
            risk_without_history=float(pooled[j]/pooled[0]),risk_ratio=float(ratio) if ratio is not None else None,
            risk_ratio_ci95=np.quantile(rr,[.025,.975]).tolist() if len(rr) else None,
            difference_pp=float((pooled[i]-pooled[j])/pooled[0]*100),
            difference_pp_ci95=(np.quantile(difference,[.025,.975])*100).tolist())
    return output


def main():
    registry=maps();tables=defaultdict(list);extras=[]
    for file in sorted((OUT/'maps').glob('*.json')):
        row=read(file)
        with np.load(row['trace_path']) as z:
            m=z['margin_db'].astype(float);up=z['network'];pulse=z['pulse']
        nt=len(m);t=np.arange(nt)
        future=np.array([a for a,b in outage_runs(up) if a>0],int)
        idx=np.searchsorted(future,t,side='right')
        lead=np.full(nt,np.inf);ok=idx<len(future)
        lead[ok]=(future[idx[ok]]-t[ok])*.5
        recent=core.rolling_sum(pulse,61)>0
        loss=core.rolling_peak(m,61)-m
        valid=(t>=60)&(t+62<nt)&up&(m>=0.)
        clean=valid&(core.rolling_sum(~up,61)==0)
        bins=np.minimum(np.floor(np.maximum(0.,m)).astype(int),60)
        for scope,eligible in (('currently_up',valid),('continuous_up30',clean)):
            for name,x in (('recent_drop30',recent),('unrestored_drop30',recent&(loss>=6.))):
                for b in np.unique(bins[eligible]):
                    a=eligible&(bins==b)&x;c=eligible&(bins==b)&~x
                    table=np.array([a.sum(),(a&(lead>0)&(lead<=30)).sum(),c.sum(),
                        (c&(lead>0)&(lead<=30)).sum(),(a&(lead>=3)&(lead<=30)).sum(),
                        (c&(lead>=3)&(lead<=30)).sum()],float)
                    tables[(scope,name)].append((row['spatial_group'],table))
        meta=read(ROOT/'outage_atlas/artifacts/maps'/(row['region']+'.json'))
        with np.load(meta['archive_path']) as z:links=z['actual_links']
        pulses=row['pulses'];frames=np.array([round(p['time_s']*2) for p in pulses],int)
        before_backup=fragile_margin(np.where(links[frames-1],1.,-1.),meta['mission']['search_drones'])>=0 if len(frames) else []
        for p,previous in zip(pulses,before_backup):
            jammer=meta['jammer'];time=p['time_s']
            interference=p['component_restoration_db']['interference']>=p['drop_db']*.5
            switched_on=(p['kind']=='intermittent' and time-.5<jammer['intermittent_start_s']<=time)
            extras.append(dict(region=row['region'],kind=p['kind'],time_s=time,
                 interference_component_explains_half=interference,
                 interference_switch_on=bool(interference and switched_on),
                 interference_receiving_path_change=bool(interference and not switched_on),
                 backup_before=bool(previous),backup_after=p['single_link_backup'],
                 new_backup_loss=bool(previous and not p['single_link_backup']),
                 onset_next30=p['first_onset_lead_s'] is not None,
                 early_onset_3to30=p['first_onset_lead_s'] is not None and p['first_onset_lead_s']>=3,
                 clean30=p['previous30_all_connected'],post_db=p['after_db'],
                 service_bad_next30=p['service_bad_next30'],service_bad_now=p['service_bad_now']))
    result=dict(created_at=now(),source_sha256=sha(OUT/'summary.json'),script_sha256=sha(__file__),
        method='Same map, 1 dB current-margin bins; each group at least 20 half-second frames; '
               'min(n_history,n_nohistory) common weights; 5000 whole spatial-group bootstrap samples',
        comparisons={scope:{feature:standardized(v,2026110509+i) for i,((s,feature),v) in enumerate(tables.items()) if s==scope}
                     for scope in ('currently_up','continuous_up30')})
    bykind={}
    for kind in ('clear','intermittent','continuous'):
        selected=[p for p in extras if p['kind']==kind]
        bykind[kind]=dict(n=len(selected),onset_next30=sum(p['onset_next30'] for p in selected),
             interference_changes=sum(p['interference_component_explains_half'] for p in selected),
             interference_switch_on=sum(p['interference_switch_on'] for p in selected),
             interference_receiving_path_change=sum(p['interference_receiving_path_change'] for p in selected))
    backup={}
    for scope in ('all','clean30'):
        selected=[p for p in extras if scope=='all' or p['clean30']]
        backup[scope]={}
        for name,f in (
             ('still_redundant',lambda p:p['backup_before'] and p['backup_after']),
             ('newly_lost_backup',lambda p:p['new_backup_loss']),
             ('already_vulnerable',lambda p:not p['backup_before'] and not p['backup_after']),
             ('newly_gained_backup',lambda p:not p['backup_before'] and p['backup_after'])):
            a=[p for p in selected if f(p)]
            backup[scope][name]=dict(n=len(a),onset_next30=sum(p['onset_next30'] for p in a),
                 early_onset_3to30=sum(p['early_onset_3to30'] for p in a),
                 post_db_median=float(np.median([p['post_db'] for p in a])) if a else None)
    result.update(component_context=bykind,backup_transitions=backup,
         no_network_onset_but_service_bad_next30=sum(not p['onset_next30'] and p['service_bad_next30'] for p in extras),
         no_network_onset_but_service_already_bad=sum(not p['onset_next30'] and p['service_bad_now'] for p in extras),
         no_network_onset_service_newly_bad=sum(not p['onset_next30'] and not p['service_bad_now'] and p['service_bad_next30'] for p in extras))
    core.save(OUT/'fine_controls.json',result)
    core.save(OUT/'pulse_context.json',extras)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
