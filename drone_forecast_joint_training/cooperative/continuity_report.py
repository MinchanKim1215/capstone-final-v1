"""v3: 원시 링크·이동·패킷 출생 코호트에서 보상과 성능을 독립 검수한다."""
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from cooperative.analyze import METRICS, key, summarize
from cooperative.continuity import (STUDY, MODELS, SEEDS, VERSION, fingerprint,
                                    protocol, verify_selection)
from cooperative.continuity_world import TERM_NAMES, REWARD_SPEC
from cooperative.refine import atomic_json
from cooperative.io import validated_result
from cooperative.refinement_analysis import (NETWORK_COLUMNS, city_for,
    trajectory_audit, paired_spatial_interval, safety_summary)
from cooperative.contracts import verify_forecaster_files
from cooperative.policy import ActorCritic
from cooperative.train import assert_rollout_policy
from swarm_sim.regions import ROOT, sha256_file

LABELS = {
    'nominal':'기존 비행', 'v2_mpc':'기존 예측 MPC',
    'v2_ppo':'기존 PPO · 위험 사용', 'v2_no_risk_ppo':'기존 PPO · 위험 제거',
    'priority_mpc':'통신 우선 MPC', 'priority_no_risk_mpc':'통신 우선 MPC · 위험 제거',
    'persistent_mpc':'통신 우선 MPC · 유지 제어', 'priority_ppo':'통신 우선 PPO',
    'priority_ppo_guarded':'통신 우선 PPO · 가드',
}
ENGLISH = {
    'nominal':'Nominal', 'v2_mpc':'Previous predictive MPC',
    'v2_ppo':'Previous PPO + risk', 'v2_no_risk_ppo':'Previous PPO, no risk',
    'priority_mpc':'Continuity MPC', 'priority_no_risk_mpc':'Continuity MPC, no risk',
    'persistent_mpc':'Continuity MPC + persistence', 'priority_ppo':'Continuity PPO',
    'priority_ppo_guarded':'Continuity PPO + guard',
}
ORDER=list(LABELS)
EXTRA_METRICS=['reward_sum','reward_mean','command_switch_count','relay_stationary_fraction']
COMPARE_METRICS=['search_connected_mean','pdr','timely_pdr','outage_total_s',
                 'outage_max_s','duration_ratio','relay_travel_m','command_switch_count']


def independent_reward_audit(z, city, row):
    """Do not call the production reward function or infer inputs from reward terms."""
    c=np.asarray(z['C'])[:,0].reshape(-1,6).astype(bool)
    n=len(c);pos=np.asarray(z['positions_xy'],float)
    cohorts=np.asarray(z['packet_cohorts'],float)
    if cohorts.shape!=(n,3) or not np.isfinite(cohorts).all() or np.any(cohorts<0):
        raise ValueError('Invalid generated/delivered/weighted birth cohort counts')
    if (np.any(cohorts[:,1]>cohorts[:,0]) or np.any(cohorts[:,2]>cohorts[:,1]+1e-8)
        or not np.array_equal(cohorts[:,:2],np.rint(cohorts[:,:2]))
        or int(cohorts[:,0].sum())!=row['generated_packets']
        or int(cohorts[:,1].sum())!=row['delivered_packets']):
        raise ValueError('Birth cohort conservation failed')
    q=.5*(cohorts[:,1]+cohorts[:,2])/np.maximum(cohorts[:,0],1)
    with np.load(ROOT/'cache/usable'/f'{row["region"]}_{row["direction"]}_motion.npz') as b:
        base=city.points[b['nodes']]
    ix=np.minimum(np.arange(n+1),len(base)-1)
    length=np.linalg.norm(np.diff(pos,axis=0),axis=-1)
    nominal_length=np.linalg.norm(np.diff(base[ix],axis=0),axis=-1)
    extra=np.maximum(0,length[:,3:]-nominal_length[:,3:]).mean(1)/15.
    targets=np.asarray(z['nominal_search_targets'],float)
    if targets.shape!=(n,3,2) or not np.isfinite(targets).all():
        raise ValueError('Nominal search targets missing')
    lag=np.linalg.norm(pos[:-1,:3]-targets,axis=-1).max(1)/40.
    request=np.asarray(z['requested_corrections'],float)
    applied=np.asarray(z['applied_corrections'],float)
    phases=np.asarray(z['phase_before_action'])
    protect=np.any(request!=0,axis=(1,2)) & (
        (np.arange(n)*3>row['baseline_duration_s']*1.25)|(lag>1.5)|(phases==2))
    expected_applied=request.copy();expected_applied[protect]=0
    if not np.array_equal(applied,expected_applied):raise ValueError('Mission protection command mismatch')
    if not np.allclose(np.asarray(z['relay_motion']),np.diff(pos,axis=0)[:,3:],rtol=0,atol=1e-12):
        raise ValueError('Actual relay displacement mismatch')
    from cooperative.candidates import OFFSETS
    actions=np.asarray(z['actions'])
    if not np.array_equal(request,OFFSETS[actions]):raise ValueError('Requested action/offset mismatch')
    switches=actions!=np.vstack([np.zeros((1,5),int),actions[:-1]])
    if not np.array_equal(switches,z['command_switches']) or int(switches.sum())!=row['command_switch_count']:
        raise ValueError('Command switch count mismatch')
    stationary=float((np.linalg.norm(np.diff(pos,axis=0)[:,3:],axis=-1)<1e-9).mean())
    if not np.isclose(stationary,row['relay_stationary_fraction'],rtol=0,atol=1e-12):
        raise ValueError('Actual stationary fraction mismatch')
    terms=np.zeros((n,8));terms[:,0]=-.5*(~c).sum(1)
    age=0.
    for i in range(n):
        for up in c[i]:
            age=0. if up else age+.5
            if not up:terms[i,1]-=.025*max(0.,min((age-1.)/5.,1.))
    terms[:,2]=-.5*(1-q)
    terms[:,3]=-.003*np.clip(extra,0,1)
    terms[:,4]=-.01*np.clip(lag,0,4)
    terms[:,5]=-.01*protect
    terms[:,6]=-.002
    if not row['mission_completed']:terms[-1,7]=-10.
    error=float(np.max(np.abs(terms-np.asarray(z['reward_terms']))))
    if error>2e-10 or not np.allclose(terms.sum(1),z['rewards'],rtol=2e-7,atol=2e-7):
        raise ValueError('Independent reward reconstruction failed')
    if not np.isclose(terms.sum(),row['reward_sum'],rtol=0,atol=1e-7):
        raise ValueError('Reward sum metadata mismatch')
    for k,value in zip(TERM_NAMES,terms.sum(0)):
        if not np.isclose(value,row['reward_terms_sum'][k],rtol=0,atol=1e-8):
            raise ValueError('Reward component metadata mismatch: '+k)
    if abs(-terms[:,0].sum()-row['outage_total_s'])>1e-9:
        raise ValueError('Outage seconds are not the primary reward cost')
    if np.any(terms[:,3]<-.003-1e-12) or np.any(terms>1e-12):
        raise ValueError('Bounded motion / nonpositive reward contract failed')
    return dict(reward_reconstructed_from_packets_and_trajectory=True,
                max_reward_term_error=error,packet_cohorts_conserved=True,
                movement_penalty_cap_verified=True,commands_recomputed=True)


def extended_summary(items):
    result=summarize(items)
    for region,values in result['by_region'].items():
        part=[r for r in items if r['region']==region]
        for k in EXTRA_METRICS:values[k]=float(np.mean([r[k] for r in part]))
        values['reward_terms_sum']={k:float(np.mean([r['reward_terms_sum'][k] for r in part])) for k in TERM_NAMES}
    for k in EXTRA_METRICS:result['macro'][k]=float(np.mean([r[k] for r in result['by_region'].values()]))
    result['macro']['reward_terms_sum']={k:float(np.mean([r['reward_terms_sum'][k] for r in result['by_region'].values()])) for k in TERM_NAMES}
    return result


def audited_episode(prefix, spec, candidate, sources, phase, tag, record=False):
    expected=sha256_file(ROOT/candidate['checkpoint']) if candidate['checkpoint'] else None
    row=validated_result(prefix,spec,expected,rollout=record)
    if (row is None or row.get('continuity_sources')!=sources
        or row.get('source_version')!=VERSION or row['controller']!=candidate['controller']
        or row['teacher_profile']!=candidate['profile'] or row.get('checkpoint_sha256')!=expected):
        raise ValueError('Raw episode/source/checkpoint contract failed: '+str(prefix))
    use_risk=candidate['controller'] not in ['nominal','reactive','mpc_no_risk','priority_mpc_no_risk'] and tag!='v2_no_risk_ppo'
    if row['risk_input_enabled']!=use_risk or (not use_risk and row['forecast_calls']!=0):
        raise ValueError('Risk input/forecaster execution contract failed')
    with np.load(Path(prefix).with_suffix('.npz')) as z:
        city=city_for(row['region'])
        audit=trajectory_audit(z,city,row)
        audit.update(independent_reward_audit(z,city,row))
        row.update({name:float(z['C'][:,column].mean()) for name,column in NETWORK_COLUMNS.items()})
    if not audit['passed'] or not row['mission_completed']:
        raise ValueError('Full-mission trajectory safety verification failed')
    return row,dict(phase=phase,tag=tag,**spec,**audit)


def training_audit(plan):
    torch.set_num_threads(1)
    all_audits=[];runs=[]
    keys=sorted(m['key'] for m in json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps'] if m['split']=='train')
    for seed in SEEDS:
        folder=MODELS/f's{seed}'
        raw_paths=[p for p in (STUDY/'training'/f's{seed}').glob('*.npz') if '.tmp.' not in p.name]
        if len(raw_paths)!=32:raise ValueError('Exactly 32 complete raw missions per v3 training seed required')
        first=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        parent=plan['parents'][str(seed)]
        if sha256_file(ROOT/parent['path'])!=parent['sha256']:raise ValueError('Warm-start parent changed')
        original=torch.load(ROOT/parent['path'],map_location='cpu',weights_only=False)
        if any(not torch.equal(v,original['model'][k]) for k,v in first['model'].items() if k.startswith('actor.')):
            raise ValueError('Initial actor did not match the declared v2 parent')
        if all(torch.equal(v,original['model'][k]) for k,v in first['model'].items() if k.startswith('critic.')):
            raise ValueError('Old critic was reused with a changed state/reward contract')
        history=final['history']
        if (final['iteration']!=4 or len(history)!=4 or sum(h['episodes'] for h in history)!=32
            or final['continuity_sources']!=plan['source_sha256'] or final['reward_spec']!=REWARD_SPEC
            or not all(h['complete_count']==8 for h in history)):
            raise ValueError('Incomplete or mismatched warm-start training')
        actor_change=float(np.sqrt(sum(float((final['model'][k]-v).double().square().sum()) for k,v in first['model'].items() if k.startswith('actor.'))))
        if actor_change<1e-8:raise ValueError('Actor did not change during training')
        schedule=np.random.default_rng(seed).permutation(keys).tolist();largest=0.
        for iteration in range(4):
            checkpoint=folder/f'iteration_{iteration:02d}.pt'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            net=ActorCritic();net.load_state_dict(ck['model']);net.eval()
            for j in range(8):
                k=iteration*8+j
                spec=dict(region=schedule[k%len(schedule)],direction=['east_west','north_south'][(k//3)%2],
                          kind=['continuous','intermittent','clear'][k%3],seed=seed+50000+k*113)
                stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
                prefix=STUDY/'training'/f's{seed}'/stem
                candidate=dict(controller='priority_sample',checkpoint=checkpoint.relative_to(ROOT).as_posix(),profile=plan['teacher_profile'])
                row,audit=audited_episode(prefix,spec,candidate,plan['source_sha256'],'training',f's{seed}',True)
                with np.load(prefix.with_suffix('.npz')) as z:
                    data={name:torch.from_numpy(z[name]) for name in ['context','candidates','mask','teacher','state','chosen','logp']}
                    largest=max(largest,assert_rollout_policy(net,data))
                all_audits.append(audit)
        runs.append(dict(seed=seed,episodes=32,updates=4,actor_change_l2=actor_change,
                         max_recomputed_logp_error=largest,all_complete=True,
                         initial_sha256=sha256_file(folder/'initial.pt'),checkpoint_sha256=sha256_file(folder/'last.pt'),
                         parent=parent,initial_actor_matches_parent=True,fresh_critic=True,
                         history=history))
    ck=torch.load(MODELS/'ensemble.pt',map_location='cpu',weights_only=False)
    if ck['member_sha256']!=[r['checkpoint_sha256'] for r in runs]:raise ValueError('Ensemble members changed')
    for member,run in zip(ck['ensemble_models'],runs):
        final=torch.load(MODELS/f's{run["seed"]}/last.pt',map_location='cpu',weights_only=False)['model']
        if any(not torch.equal(v,final[k]) for k,v in member.items()):raise ValueError('Embedded ensemble differs from member weights')
    return runs,all_audits


def phase_audit(plan,phase):
    groups={};audits=[]
    expected={key(s) for s in plan[phase]}
    for tag,candidate in plan['candidates'].items():
        raw=json.loads((STUDY/phase/f'{tag}_complete.json').read_text(encoding='utf-8'))
        if raw['candidate']!=candidate or len(raw['results'])!=len(expected) or {key(r) for r in raw['results']}!=expected:
            raise ValueError('Prespecified method/scenario set mismatch: '+phase+'/'+tag)
        rows=[]
        for spec in plan[phase]:
            stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
            row,audit=audited_episode(STUDY/phase/tag/stem,spec,candidate,plan['source_sha256'],phase,tag)
            declared=next(r for r in raw['results'] if key(r)==key(spec))
            for name in ['array_sha256','checkpoint_sha256','search_connected_mean','pdr','reward_sum']:
                if declared.get(name)!=row.get(name):raise ValueError('Aggregate and raw sidecar differ: '+name)
            rows.append(row);audits.append(audit)
        groups[tag]=sorted(rows,key=key)
    return groups,audits


def pilot_audit(plan):
    pilot=json.loads((STUDY/'pilot_plan.json').read_text(encoding='utf-8'))
    if pilot['source_sha256']!=plan['source_sha256'] or not pilot['training_regions_only']:
        raise ValueError('Training-only pilot contract failed')
    audits=[]
    for tag,candidate in pilot['candidates'].items():
        for spec in pilot['specs']:
            stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
            _,audit=audited_episode(STUDY/'pilot'/tag/stem,spec,candidate,plan['source_sha256'],'training_pilot',tag)
            audits.append(audit)
    if len(audits)!=48:raise ValueError('48 training-only full pilot missions required')
    return audits


def analyze():
    plan=protocol();freeze=verify_selection(plan)
    if fingerprint()!=plan['source_sha256'] or verify_forecaster_files()!=plan['forecaster_files']:
        raise ValueError('Frozen source/forecaster contract changed')
    training,train_audits=training_audit(plan)
    validation,val_audits=phase_audit(plan,'validation')
    groups,test_audits=phase_audit(plan,'test')
    pilot_audits=pilot_audit(plan)
    # Independently recompute the validation choice from the raw sidecars.
    vs={tag:extended_summary(rows) for tag,rows in validation.items()}
    base=vs['nominal']['macro']['pdr']
    feasible={k:v['complete_count']==28 and v['safety_violations']==0 and v['max_duration_ratio']<=1.15
              and v['macro']['pdr']>=base-.005 for k,v in vs.items()}
    selected=min([k for k in vs if feasible[k]],key=lambda k:(-vs[k]['macro']['search_connected_mean'],
                 -vs[k]['macro']['timely_pdr'],vs[k]['macro']['relay_travel_m']))
    if selected!=freeze['selected'] or feasible!=freeze['feasible']:raise ValueError('Validation choice cannot be reproduced')
    summaries={tag:extended_summary(rows) for tag,rows in groups.items()}
    maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    map_groups={m['key']:m['spatial_group'] for m in maps}
    def compare(a,b):return {m:paired_spatial_interval(groups[a],groups[b],m,map_groups) for m in COMPARE_METRICS}
    vs_nominal={tag:compare(tag,'nominal') for tag in groups if tag!='nominal'}
    vs_previous_mpc={tag:compare(tag,'v2_mpc') for tag in groups if tag!='v2_mpc'}
    comparisons={
        'priority_mpc_risk_vs_no_risk':compare('priority_mpc','priority_no_risk_mpc'),
        'priority_ppo_vs_priority_mpc':compare('priority_ppo','priority_mpc'),
        'guarded_ppo_vs_priority_mpc':compare('priority_ppo_guarded','priority_mpc'),
        'persistence_vs_priority_mpc':compare('persistent_mpc','priority_mpc'),
        'previous_ppo_risk_vs_no_risk':compare('v2_ppo','v2_no_risk_ppo'),
    }
    records=[]
    for tag,rows in groups.items():
        for row in rows:
            record=dict(method=tag,spatial_group=map_groups[row['region']],**row)
            record.update({'cost_'+k:row['reward_terms_sum'][k] for k in TERM_NAMES})
            records.append(record)
    fields=['method','spatial_group','region','direction','kind','seed','mission_completed',*METRICS,
            *NETWORK_COLUMNS,*EXTRA_METRICS,*['cost_'+k for k in TERM_NAMES],
            'generated_packets','delivered_packets','ttl_drops','retry_drops','overflow_drops',
            'retries','beacon_tx','beacon_rx','forecast_calls','risk_input_enabled','controller',
            'teacher_profile','guard_regret','reward_objective','source_version','checkpoint_sha256','array_sha256']
    fields=list(dict.fromkeys(fields))
    with (STUDY/'comparison_all_metrics_v3.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    formal=train_audits+val_audits+test_audits
    if (len(train_audits),len(val_audits),len(test_audits),len(formal))!=(96,252,288,636):
        raise ValueError('Required 96 training +252 validation +288 test missions missing')
    result=dict(format='communications-priority-audited-v3',selected_by_validation=selected,
                selected_policy=freeze['selected_policy'],teacher_profile=plan['teacher_profile'],
                counts=dict(training=96,validation=252,test=288,formal=636,pilot=48),
                summaries=summaries,validation_summaries=vs,vs_nominal=vs_nominal,
                vs_previous_mpc=vs_previous_mpc,comparisons=comparisons,training_audit=training,
                safety=safety_summary(test_audits),formal_safety=safety_summary(formal),pilot_safety=safety_summary(pilot_audits),
                reward_audit=dict(all_reconstructed=True,max_term_error=max(a['max_reward_term_error'] for a in formal),
                                  raw_inputs=['links','packet_cohorts','positions_xy','nominal_search_targets',
                                              'requested_corrections','phase_before_action','baseline motion cache']),
                selection_sha256=sha256_file(STUDY/'selection.json'),source_sha256=plan['source_sha256'],
                forecaster_files_sha256=plan['forecaster_files'],
                completed_test_sha256={f.name:sha256_file(f) for f in sorted((STUDY/'test').glob('*_complete.json'))},
                regions=8,independent_spatial_groups=6,test_used_for_selection=False,
                test_scope=plan['test_scope'],
                limitations=['same geographic test maps as previous versions, fresh scenario seeds',
                    'spatial-group intervals condition on the fitted ensembles; training-seed uncertainty is not included',
                    'warm-start PPO reward and teacher changed together; not a pure reward-only causal ablation',
                    'three additional PPO seeds, 32 new full missions each; no convergence or optimum guarantee',
                    'no separately retrained v3 no-risk PPO; no claim of causal risk benefit in new PPO',
                    'shared ideal routing/safety and fixed LightGBM; no hardware or full 3GPP validation'])
    atomic_json(STUDY/'training_trajectory_audit.json',train_audits)
    atomic_json(STUDY/'validation_trajectory_audit.json',val_audits)
    atomic_json(STUDY/'trajectory_audit.json',test_audits)
    atomic_json(STUDY/'all_formal_trajectory_audit.json',formal)
    atomic_json(STUDY/'pilot_trajectory_audit.json',pilot_audits)
    atomic_json(STUDY/'audited_summary.json',result)
    print(json.dumps(dict(audited=len(formal),pilot=len(pilot_audits),selected=selected,
                         reward_reconstruction=result['reward_audit'],safety=result['formal_safety']),ensure_ascii=False),flush=True)
    return result


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'docs/assets/continuity';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    paths={}
    fig,ax=plt.subplots(figsize=(9.4,5.3),layout='constrained')
    tags=[k for k in ORDER if k!='nominal']
    for i,k in enumerate(tags):
        r=data['vs_nominal'][k]['search_connected_mean']
        color='#077d8c' if k==data['selected_by_validation'] else '#76899e'
        ax.hlines(i,100*r['ci95_low'],100*r['ci95_high'],color=color,lw=2.5)
        ax.scatter(100*r['difference'],i,s=42,color=color,zorder=3)
    ax.set_yticks(range(len(tags)),[ENGLISH[k] for k in tags]);ax.invert_yaxis()
    ax.axvline(0,color='#b05c41',ls='--',lw=1);ax.grid(axis='x',alpha=.18)
    ax.set_xlabel('Search connectivity change vs nominal (percentage points)')
    ax.set_title('Fresh paired scenarios · 32 per policy · 6 spatial groups',loc='left',pad=12)
    p=folder/'connectivity_intervals.png';fig.savefig(p,dpi=160);plt.close(fig);paths['intervals']=p
    fig,ax=plt.subplots(figsize=(9.3,4.7),layout='constrained')
    x=np.arange(len(ORDER));bottom=np.zeros(len(ORDER))
    colors=['#0a6c83','#4c91ab','#e3b364','#9ab6a1','#c0b3ca','#db936f','#c2cad2','#b75952']
    labels=['Outage seconds','Long outage excess','Packet quality','Extra movement','Search lag','Mission protection','Time','Failure']
    for name,color,label in zip(TERM_NAMES,colors,labels):
        y=np.array([-data['summaries'][k]['macro']['reward_terms_sum'][name] for k in ORDER])
        ax.bar(x,y,bottom=bottom,width=.68,color=color,label=label);bottom+=y
    ax.set_xticks(x,[ENGLISH[k] for k in ORDER],rotation=32,ha='right',fontsize=8)
    ax.set_ylabel('Mean mission cost (lower is better within this objective)')
    ax.set_title('Reward decomposition · one common v3 objective for all policies',loc='left',pad=12)
    ax.legend(ncol=4,fontsize=7,loc='upper left');ax.grid(axis='y',alpha=.15)
    p=folder/'reward_costs.png';fig.savefig(p,dpi=160);plt.close(fig);paths['costs']=p
    return paths


def _uri(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
def _num(x,n=3):return '—' if x is None else f'{x:,.{n}f}'
def _interval(r):return f'{r["difference"]*100:+.3f}%p [{r["ci95_low"]*100:+.3f}, {r["ci95_high"]*100:+.3f}]'


def build():
    from cooperative.report import font_css
    data=json.loads((STUDY/'audited_summary.json').read_text(encoding='utf-8'))
    freeze=verify_selection(protocol());paths=figures(data)
    selected=freeze['selected'];m=data['summaries'][selected]['macro'];b=data['summaries']['nominal']['macro']
    best=max(ORDER,key=lambda k:data['summaries'][k]['macro']['search_connected_mean'])
    rows=[]
    for k in ORDER:
        s=data['summaries'][k];r=s['macro']
        rows.append('<tr'+(' class="chosen"' if k==selected else '')+'>'+f'<th>{LABELS[k]}'+(' <small>검증 선정</small>' if k==selected else '')+'</th>'+
            f'<td>{r["search_connected_mean"]*100:.3f}%</td><td>{r["pdr"]*100:.3f}%</td><td>{r["timely_pdr"]*100:.3f}%</td>'+
            f'<td>{r["outage_total_s"]:.1f}</td><td>{r["outage_max_s"]:.1f}</td><td>{r["duration_ratio"]:.3f}</td>'+
            f'<td>{r["relay_travel_m"]/1000:.2f}</td><td>{r["command_switch_count"]:.0f}</td><td>{s["complete_count"]}/32</td></tr>')
    comparisons=[]
    for k,label in [('priority_mpc_risk_vs_no_risk','새 MPC: 예측 위험 입력 효과'),
                    ('priority_ppo_vs_priority_mpc','새 PPO − 새 MPC'),('guarded_ppo_vs_priority_mpc','가드 PPO − 새 MPC'),
                    ('persistence_vs_priority_mpc','유지 제어 − 새 MPC'),('previous_ppo_risk_vs_no_risk','기존 PPO: 위험 사용 − 제거')]:
        r=data['comparisons'][k]['search_connected_mean']
        conclusion='평균 개선 관측' if r['ci95_low']>0 else '평균 저하 관측' if r['ci95_high']<0 else '구간이 0을 포함'
        comparisons.append(f'<tr><th>{label}</th><td>{_interval(r)}</td><td>{conclusion}</td></tr>')
    seeds=''.join(f'<tr><td>{r["seed"]}</td><td>{r["episodes"]}</td><td>{r["updates"]}</td><td>{r["actor_change_l2"]:.4f}</td><td>{r["max_recomputed_logp_error"]:.2e}</td></tr>' for r in data['training_audit'])
    validation_rows=''.join(f'<tr><th>{LABELS[k]}</th><td>{data["validation_summaries"][k]["macro"]["search_connected_mean"]*100:.3f}%</td><td>{data["validation_summaries"][k]["macro"]["pdr"]*100:.3f}%</td><td>{"통과" if freeze["feasible"][k] else "제외"}</td></tr>' for k in ORDER)
    pilot=json.loads((STUDY/'pilot_selection.json').read_text(encoding='utf-8'))
    pilot_labels={'v2_mpc':'기존 MPC','linear':'선형 점수','probability':'확률 점수','persistent':'선형 점수 + 유지 제어'}
    pilot_rows=''.join(f'<tr><th>{pilot_labels[k]}</th><td>{v["macro"]["search_connected_mean"]*100:.3f}%</td><td>{v["macro"]["pdr"]*100:.3f}%</td><td>{v["max_duration_ratio"]:.3f}</td><td>{v["complete_count"]}/12</td></tr>' for k,v in pilot['summaries'].items())
    safe=data['formal_safety'];delta=data['vs_nominal'][selected]['search_connected_mean'] if selected!='nominal' else None
    lead_effect=_interval(delta) if delta else '기준 비행과 동일'
    css='''*{box-sizing:border-box}body{margin:0;background:#edf3f5;color:#17313e;font:16px/1.75 ReportKR,"Malgun Gothic",Arial,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px 70px}header{padding:32px;border-radius:20px;background:#123643;color:white}header p{color:#d1e4eb}h1{font-size:34px;line-height:1.35;margin:8px 0 18px}h2{font-size:23px;margin:0 0 16px}h3{font-size:18px}section{padding:28px;margin-top:22px;background:white;border:1px solid #d9e4e8;border-radius:16px}.tag{letter-spacing:.13em;font-size:12px;color:#9ee2e3}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:22px}.metric{padding:18px;background:#e9f5f5;border-radius:12px}.metric strong{display:block;font-size:29px;color:#086c7a}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid #dfe8ec;text-align:right;white-space:nowrap}th:first-child{text-align:left}thead{background:#edf4f6}.chosen{background:#e4f5f1}small{color:#08786e}img{display:block;width:100%;height:auto;margin:20px auto}.note{padding:16px 20px;background:#fff5e6;border-left:4px solid #d69b39;border-radius:5px}code{background:#edf2f5;padding:2px 5px;border-radius:4px;font-size:13px;overflow-wrap:anywhere}pre{background:#122f3a;color:#e5f3f7;padding:18px;border-radius:10px;overflow:auto}pre code{background:none;color:inherit}details{margin:14px 0}summary{cursor:pointer;font-weight:700}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}footer{font-size:12px;color:#526d7a;padding-top:24px;overflow-wrap:anywhere}@media(max-width:800px){.metrics,.grid{grid-template-columns:1fr}main{padding:14px}section,header{padding:20px}h1{font-size:27px}}'''
    text='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>통신 우선 학습·재배치 v3 검수 결과</title><style>__FONT____CSS__</style></head><body><main>
<header><div class="tag">CAPSTONE DESIGN 2 · COMMUNICATIONS PRIORITY · V3</div><h1>단절 시간을 줄이는<br>학습·재배치 통합 결과</h1><p>교수님 피드백과 첨부 MADDPG를 검토한 뒤, 기존 물리 엔진을 보존하고 별도 보상·제어·학습을 실제 실행했다. 기본값은 검증 지역에서 선정한 __SELECTED__이다.</p></header>
<div class="metrics"><div class="metric">수색군 연결 유지율<strong>__CTR__%</strong>기존 비행 __BASE_CTR__%</div><div class="metric">실제 패킷 전달률<strong>__PDR__%</strong>기존 비행 __BASE_PDR__%</div><div class="metric">공식 전체 임무 검수<strong>636 / 636</strong>별도 학습용 후보 비교 48회 · 시연 6회</div></div>
<section><h2>1. 먼저 읽을 결론</h2><p>검증 선정 방식의 연결률 변화는 __EFFECT__이다. 8개 지도에 동일한 새 난수 32조건을 대응 비교하고, 인접·중첩 지도는 6개 공간 그룹으로 묶어 재표집했다. 구간은 적은 독립 그룹에서 얻은 설명적 구간이다.</p><p>이번 시험의 연결률 수치상 최고는 <strong>__BEST__ (__BEST_CTR__%)</strong>다. 시험 전에 선정한 기본값과 구분하며 시험 결과로 기본값을 교체하지 않았다. 이전 v2의 다른 난수 평균과 직접 증감 비교하지 않고, 기존 MPC·PPO도 이번 조건에서 다시 실행했다.</p><p class="note">보상합의 크기는 성능 판정 기준이 아니다. 기본값은 완주·안전·PDR·임무 시간 제한을 먼저 통과하고, 검증 연결률 → 적시 전달률 → 중계 이동량 순서로 선정했다. 학습 알고리즘 이름만으로 성능 우위를 전제하지 않았다.</p></section>
<section><h2>2. 교수님 피드백과 첨부 코드 반영</h2><div class="table"><table><thead><tr><th>검토 사항</th><th>반영한 설계와 확인</th></tr></thead><tbody>
<tr><th>누적 보상은 스케일에 의존</th><td>연결률·PDR·단절 초·완주·이동량을 별도로 비교. 보상 8항을 원시 기록에서 재계산.</td></tr>
<tr><th>통신 단절 예방 우선, 이동 비용은 작게</th><td>단절 1초 = 비용 1. 한 관측의 0.5초 단절 비용은 최대 이동 비용 0.003의 166.7배.</td></tr>
<tr><th>긴 단절과 패킷 전달</th><td>긴 연속 단절에 제한된 추가 비용, 생성 시점별 패킷 코호트의 최종 전달·지연 품질 정산.</td></tr>
<tr><th>MADDPG의 폐루프·CTDE</th><td>폐루프 통합·학습 전용 중앙 Critic 유지. 현재 입력 계약에 맞는 공유 PPO를 별도 재학습.</td></tr>
<tr><th>MADDPG의 유지 제어</th><td>작은 전환 비용과 별도 히스테리시스 후보 비교. 명목 복귀·이전 명령 유지·새 행동을 구분.</td></tr>
<tr><th>첨부 lookahead / 과도한 H</th><td>실제 미래 계획·간섭 시각 정답과 이상적 전역 상태를 Actor에 도입하지 않음. 보정 요청 H를 큰 벌점으로 쓰지 않음.</td></tr>
<tr><th>기존 안전 이동</th><td>연속 5m 분리·건물 금지·그래프 이동·5m/s 상한을 공통 이동기로 적용. 학습 보상으로 안전을 보장했다고 주장하지 않음.</td></tr>
</tbody></table></div><p>첨부 실행 CSV를 다시 계산하면 기존 비행은 연결률 93.76194%, 보상합 355.95676이고, MADDPG L2 0.05는 96.78508%, 보상합 215.45220이다. 후자는 통신이 좋아져도 H 비용 −141.75 때문에 보상합이 낮다. 이 수치는 첨부의 별도 지도·시나리오 결과이며 본 시험과 직접 성능 비교할 수 없다.</p><p>첨부 모델은 관측 112차원·연속 2D 행동이고 현재 정책은 문맥 91차원·후보별 12특징·25개 이동 후보여서 가중치를 직접 연결하지 않았다. 상세 근거는 <code>docs/MADDPG_교수피드백_반영검토.md</code>와 <code>docs/maddpg_review/attachment_reward_recalculation.json</code>에 있다.</p></section>
<section><h2>3. 같은 조건에서 실제 비교한 성능</h2><p>지역별 평균을 다시 동일 가중치로 평균했다. 연결 유지율은 수색 3대가 중계 경유를 포함해 모두 상호 도달 가능한 0.5초 구간의 비율이다. 명령 전환 수는 드론별 후보 인덱스 변경 합이며 실제 비행 거리와 구분한다.</p><div class="table"><table><thead><tr><th>방식</th><th>연결률</th><th>PDR</th><th>적시 PDR</th><th>총 단절 평균(s)</th><th>최장 단절 평균(s)</th><th>시간 배율</th><th>중계 이동(km)</th><th>명령 전환</th><th>완주</th></tr></thead><tbody>__ROWS__</tbody></table></div><img alt="동일 난수 시험의 연결률 변화와 공간 그룹 재표집 구간" src="__INTERVALS__"><div class="table"><table><thead><tr><th>대응 비교</th><th>연결률 차이와 95% 구간</th><th>해석</th></tr></thead><tbody>__COMPARISONS__</tbody></table></div><p>새 PPO는 위험 제거군을 별도로 재학습하지 않았으므로 새 PPO에서 예측 위험만의 인과적 추가 효과는 주장하지 않는다. 새 보상과 지도 점수를 함께 바꿨으므로 기존 대비 변화도 보상 하나만의 효과로 해석할 수 없다.</p></section>
<section><h2>4. 통신 우선 보상과 학습</h2><p>3초 제어 구간 비용 = 단절 초 + 긴 단절 추가 비용 + 0.5(1−패킷 품질) + 최대 0.003 추가 이동 + 최대 0.04 수색 지연 + 0.01 임무 보호 + 0.002 시간 + 미완주 시 10. 모든 항은 비용이므로 불필요한 임무 연장으로 양의 생존 보상을 모을 수 없다. 패킷 품질은 해당 구간에서 생성된 패킷이 최종 정산된 뒤 계산한다.</p><img alt="동일 v3 목적식으로 다시 평가한 방식별 비용 분해" src="__COSTS__"><div class="grid"><div><h3>추가 학습</h3><p>학습 지도에서 48개 전체 임무로 지도 점수 형식만 고르고, 세 v2 위험 사용 Actor를 각각 초기값으로 사용했다. Critic과 Adam은 새로 초기화했다. 독립 난수 3개 × 새 전체 임무 32개 = 96회, 각 정책 4회 갱신, 세 정책 동일 가중치 logit 앙상블이다. 고정 LightGBM은 역전파하지 않는다.</p></div><div><h3>시험 전 선정</h3><p>학습 30 / 검증 7 / 시험 8 지도, 방향 2개와 간섭 2종의 균형 조건. 검증 28조건 ×9방식=252회, 새 난수 시험 32조건 ×9방식=288회다. 기본값 __SELECTED__은 모든 검증 완주·실제 안전 위반 0·PDR 기준 대비 −0.5%p 이내·임무 시간 최대 1.15배를 만족한 후보 중 선정했다.</p></div></div><details><summary>추가 학습 확인</summary><div class="table"><table><thead><tr><th>학습 난수</th><th>새 전체 임무</th><th>갱신</th><th>Actor 변화 L2</th><th>재계산 old logp 오차</th></tr></thead><tbody>__SEEDS__</tbody></table></div><p>초기 Actor가 선언한 부모와 같은지, Critic이 새로 초기화됐는지, 각 수집 당시 체크포인트로 저장한 모든 행동의 logp를 다시 계산하는지 확인했다. 32회 추가 학습은 최적 수렴을 증명하지 않는다.</p></details><details><summary>검증 지역 선정 표</summary><div class="table"><table><thead><tr><th>방식</th><th>연결률</th><th>PDR</th><th>제약</th></tr></thead><tbody>__VALIDATION__</tbody></table></div></details></section>
<section><h2>5. 검수와 실제 적용 범위</h2><p>공식 636개 임무의 저장 궤적·링크·패킷 보존·명령·보상 항을 별도로 검사했다. 최소 연속 이동 분리 <strong>__MIN_SEP__m</strong>, 충돌·건물 침범·잘못된 그래프 이동 0건이다. 각 무선 링크 행렬에서 전체 10개 네트워크 지표를 다시 계산했다. 보상 항 최대 재계산 오차는 __REWARD_ERROR__다.</p><ul><li>전체 8대(수색3/중계5), V40m, 400m×400m, 고도50m. RF 0.5초·제어3초·25개 이동 후보.</li><li>통신 위험 예측은 실제 수신 비콘·ACK·이력 기반. 행동 입력에는 실제 미래 궤적·미래 간섭 시각을 사용하지 않는다. 중앙 Critic은 학습 전용이다.</li><li>ITU FSPL·Bullington 회절 근사·시간 상관 섀도잉·외부 간섭·BPSK·이상적 TDMA. 3GPP TR36.777 전체 구현은 아니다.</li><li>수색 드론 간 전달이 목적이며 GCS 목적지·실제 기체의 조종 통신은 포함하지 않는다.</li><li>이상적 전역 라우팅·공동 안전 이동·동기화·계산 지연 없는 시뮬레이션 시간 가정이 있다. 실제 기체·Wi-Fi 스택·전력·GPS 오차·바람·계산 지연·분산 안전 통신 검증은 별도다.</li><li>기존 v1/v2에서 시험했던 지도 8곳에 새 난수를 사용했다. 새로운 지리 영역의 최초 독립 외부 검증으로 해석할 수 없다.</li><li>HTML 내용·데이터·프레임·JavaScript 정적 검수는 수행하되 브라우저 파일 접근 정책으로 실제 상호작용 UI 검수는 수행하지 못했다.</li></ul></section>
<section><h2>6. 실행과 확인할 파일</h2><pre><code>python cooperate_run.py doctor
python cooperate_run.py verify
python cooperate_run.py compare --region gangnam --seed 2026 --out results/continuity/custom/gangnam
python cooperate_run.py continuity --stage all --workers 4</code></pre><p>새 결과: <code>results/continuity/selection.json</code>, <code>audited_summary.json</code>, <code>comparison_all_metrics_v3.csv</code>, <code>all_formal_trajectory_audit.json</code>. 새 모델: <code>model/continuity/ensemble.pt</code>. 이전 v2 실험과 핵심 소스는 보존했다. 개별 연구 조건은 <code>configs/continuity_plan.json</code>에 고정했다.</p><p>방법 근거: <a href="https://arxiv.org/abs/1707.06347">PPO 원 논문</a>, <a href="https://arxiv.org/abs/1706.02275">MADDPG 원 논문</a>. 알고리즘 이름에 의한 보편적 우위·수렴·실기체 안전 보장은 주장하지 않는다.</p></section>
<footer>v3 검증 선정 SHA256: __SELECTION_SHA__<br>예측기 파일과 원시 결과 지문은 audited_summary.json에 기록했다.</footer></main></body></html>'''
    text=text.replace('<details><summary>추가 학습 확인</summary>',
        '<details><summary>학습 지역에서의 후보 비교 48회</summary><p>세 학습 지역 × 방향2 × 간섭2의12조건에서 네 구성을 비교했다. PPO 교사점수는 선형/확률 중 '+pilot_labels[pilot['teacher_profile']]+'를 고르고, 유지 제어는 학습용 비교와 동일한 선형 점수+히스테리시스로 별도 평가했다.</p><div class="table"><table><thead><tr><th>구성</th><th>연결률</th><th>PDR</th><th>시간 최대 배율</th><th>완주</th></tr></thead><tbody>'+pilot_rows+'</tbody></table></div></details><details><summary>추가 학습 확인</summary>')
    text=text.replace('구간은 적은 독립 그룹에서 얻은 설명적 구간이다.',
        '구간은 적은 독립 그룹에서 얻은 설명적 구간이며, 학습 난수를 다시 바꿨을 때의 모델 성능 분산까지 반영하지 않는다.')
    values={'__FONT__':font_css(text+css+''.join(rows)+''.join(comparisons)+validation_rows),'__CSS__':css,'__SELECTED__':LABELS[selected],
            '__CTR__':f'{m["search_connected_mean"]*100:.3f}','__PDR__':f'{m["pdr"]*100:.3f}',
            '__BASE_CTR__':f'{b["search_connected_mean"]*100:.3f}','__BASE_PDR__':f'{b["pdr"]*100:.3f}',
            '__EFFECT__':lead_effect,'__BEST__':LABELS[best],'__BEST_CTR__':f'{data["summaries"][best]["macro"]["search_connected_mean"]*100:.3f}',
            '__ROWS__':''.join(rows),'__COMPARISONS__':''.join(comparisons),'__SEEDS__':seeds,
            '__VALIDATION__':validation_rows,'__INTERVALS__':_uri(paths['intervals']),'__COSTS__':_uri(paths['costs']),
            '__MIN_SEP__':f'{safe["minimum_swept_separation_m"]:.3f}',
            '__REWARD_ERROR__':f'{data["reward_audit"]["max_term_error"]:.2e}',
            '__SELECTION_SHA__':data['selection_sha256']}
    for a,bvalue in values.items():text=text.replace(a,bvalue)
    path=ROOT/'docs/통신우선_결과_v3.html';path.write_text(text,encoding='utf-8');return path


if __name__=='__main__':analyze();print(build())
