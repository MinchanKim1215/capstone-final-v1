"""Selection-only RF-equation attribution; never a model input or test score."""
from collections import Counter,defaultdict
import csv
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,digest,now,V9


def main():
    from adaptive_joint_expanded.world import World
    from cooperative.packets import hop_probabilities
    from swarm_sim.radio import packet_success
    from forecast_rebuild.dataset import Episode
    registry={r['key']:r for r in read(V9/'dataset.json')['rows'] if r['role']=='selection'}
    grouped=defaultdict(list)
    with (ARTIFACTS/'diagnostics/events.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):grouped[row['case']].append(row)
    samples=[];dominant=Counter();prevent=Counter();maximum=0.
    for count,(key,events) in enumerate(grouped.items()):
        record=registry[key];ep=Episode(record);env=World(**record['spec'],deployment=record['deployment'],observer=None,max_steps=1)
        with np.load(record['array_path'],allow_pickle=False) as a:positions=a['positions']
        selected=sorted(events,key=lambda r:digest(['physics-diagnostic-v10',key,r['observer'],r['peer'],r['onset_s']]))[:12]
        def state(q):
            k=q//6;f=(q%6)/6;xy=positions[k]*(1-f)+positions[k+1]*f;xyz=np.c_[xy,np.full(len(xy),50.)]
            tx=np.repeat(xyz,len(xyz),axis=0);rx=np.tile(xyz,(len(xyz),1));loss=env.city.radio.propagation(tx,rx,env.scenario)[0].reshape(len(xyz),len(xyz))
            jl=env.city.radio.propagation(env.jammer[None],xyz,env.scenario)[0]
            interference=10**((env.jam_power-jl)/10)*bool(env.jam_start<=q*.5<env.jam_end)
            denominator=10*np.log10(10**(-107/10)+interference);received=np.empty_like(loss);sinr=np.empty_like(loss)
            p=hop_probabilities(loss,interference,env.shadow[q],received,sinr)
            np.testing.assert_equal((p>=.9)&(p.T>=.9),ep.links[q])
            return loss,np.array(env.shadow[q]),denominator,sinr
        for row in selected:
            q=int(round(float(row['onset_s'])*2));i=int(row['observer']);j=int(row['peer']);before=state(q-1);after=state(q)
            if after[3][j,i]<after[3][i,j]:i,j=j,i
            changes=dict(geometry_path_loss=float(after[0][i,j]-before[0][i,j]),random_shadow=float(after[1][i,j]-before[1][i,j]),
                interference_plus_noise=float(after[2][j]-before[2][j]))
            error=abs((after[3][i,j]-before[3][i,j])+sum(changes.values()));maximum=max(maximum,error);assert error<1e-9
            main=max(changes,key=changes.get);dominant[main]+=1;preventions={}
            for component in changes:
                s1=after[3][i,j];s2=after[3][j,i]
                if component=='geometry_path_loss':s1+=after[0][i,j]-before[0][i,j];s2+=after[0][j,i]-before[0][j,i]
                elif component=='random_shadow':s1+=after[1][i,j]-before[1][i,j];s2+=after[1][j,i]-before[1][j,i]
                else:s1+=after[2][j]-before[2][j];s2+=after[2][i]-before[2][i]
                p1=packet_success(s1,4608,1e6,1e6)*packet_success(s2,112,1e6,1e6)
                p2=packet_success(s2,4608,1e6,1e6)*packet_success(s1,112,1e6,1e6)
                restored=bool(p1>=.9 and p2>=.9);preventions[component]=restored;prevent[component]+=int(restored)
            samples.append(dict(case=key,onset_s=float(row['onset_s']),duration_s=float(row['duration_s']),strict_hit=row['strict_hit']=='True',
                largest_adverse_component=main,delta_components_db=changes,hold_previous_component_prevents_onset=preventions))
        write(ARTIFACTS/'physics_diagnosis_status.json',dict(complete=count+1,total=len(grouped),sampled_events=len(samples),updated_at=now()))
    write(ARTIFACTS/'diagnostics/physics_components.json',dict(created_at=now(),scope='Selection only; deterministic sample of at most12 onsets per flight',
        sampled_events=len(samples),flights=len(grouped),largest_adverse_component=dict(dominant),hold_previous_component_prevents_onset=dict(prevent),
        exact_sinr_decomposition_max_error_db=maximum,samples=samples,
        limitation='Within this simulator equation over the last half second only. Counterfactual prevention flags can overlap; this is not a population-cause estimate or a 3-second predictability bound. No model/threshold changes use these results.'))
    print(ARTIFACTS/'diagnostics/physics_components.json')


if __name__=='__main__':main()
