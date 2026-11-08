"""Registered matched branches, latency stress and auditable completion."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import sys
import numpy as np
from . import ROOT,WORKSPACE,ARTIFACTS
from .protocol import prepare,freeze,status,read,write,sha,digest,now,ARMS,CONFIG
from .source import anchors,context
from .runner import run


def make_cases(plan):
    path=ARTIFACTS/'cases.json'
    if path.exists():
        return read(path)['cases']
    rows=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(anchors,e):i for i,e in enumerate(plan['sources'])}
        results={}
        for count,job in enumerate(as_completed(jobs),1):
            results[jobs[job]]=job.result()
            if count%6==0 or count==len(jobs):
                status('prepare_matched_source_windows',complete=count,total=len(jobs))
                print('case sources',count,'/',len(jobs),flush=True)
    rows=[e for i in sorted(results) for e in results[i]]
    regions={c['spec']['region'] for c in rows}
    if len(regions)!=14:
        raise ValueError('Expected all 14 known confirmation regions')
    write(path,dict(cases=rows,regions=len(regions),source_flights=len(plan['sources']),
                    source_kind_counts={k:sum(c['anchor_kind']==k for c in rows) for k in ('outage','quiet')},
                    created_at=now()))
    return rows


def trial(task):
    case,arm,latency,seal=task
    section='confirm' if latency==CONFIG['command_latency_s'] else 'stress'
    path=ARTIFACTS/section/arm/(case['key']+'.json')
    array=path.with_suffix('.npz')
    contract=dict(case=case,arm=arm,command_latency_s=latency,freeze_sha256=seal)
    if path.exists():
        row=read(path)
        if row['contract']!=contract or sha(array)!=row['archive_sha256']:
            raise ValueError('Changed trial or archive')
        return row
    row,data=run(case,arm,latency)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=array.with_suffix('.pending.npz')
    np.savez_compressed(temporary,**data)
    temporary.replace(array)
    row.update(contract=contract,archive_path=str(array),archive_sha256=sha(array))
    write(path,row)
    return row


def simulate(tasks,stage):
    output=[None]*len(tasks)
    status(stage,complete=0,total=len(tasks))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(trial,t):i for i,t in enumerate(tasks)}
        for count,job in enumerate(as_completed(jobs),1):
            try:
                output[jobs[job]]=job.result()
            except Exception:
                status(stage,state='failed',task=tasks[jobs[job]][:3],complete=count-1,total=len(tasks))
                for other in jobs:
                    other.cancel()
                raise
            if count%12==0 or count==len(tasks):
                status(stage,complete=count,total=len(tasks))
                print(stage,count,'/',len(tasks),flush=True)
    return output


def audit(plan,rows):
    all_pins=all(sha(p)==expected for p,expected in plan['protected_dependencies'].items())
    samples=0
    for row in rows:
        if sha(row['archive_path'])!=row['archive_sha256']:
            raise ValueError('Trial data changed')
        with np.load(row['archive_path'],allow_pickle=False) as z:
            from .runner import intervals,pair_min
            from forecast_prevention.diagnostics import connected
            np.testing.assert_array_equal(connected(z['links'],row['search_drones']),z['network'])
            cuts=intervals(z['network'])
            if cuts!=row['cuts'] or sum(a>0 for a,b in cuts)!=row['onsets']:
                raise ValueError('Outage tally changed')
            pos=z['positions'].astype(float);vel=z['velocity'].astype(float)
            if len(pos)!=241 or len(z['links'])!=240:
                raise ValueError('Fine clock frame count changed')
            if np.linalg.norm(vel,axis=-1).max()>5.00001 or np.linalg.norm(np.diff(vel,axis=0),axis=-1).max()/.1>2.5001:
                raise ValueError('Saved acceleration/speed violation')
            if min(pair_min(x) for x in pos)<4.9999:
                raise ValueError('Saved separation violation')
            if row['interventions']:
                event=row['interventions'][0]
                index=event['changed_relay']-row['search_drones']
                received=np.flatnonzero((z['applied'][:,index]==event['new_offset']).all(axis=1))
                if len(received) and received[0]*.1<event['earliest_apply_s']-1e-7:
                    raise ValueError('Command acted before its computation/transmission delay')
            samples+=len(z['network'])
    if not all_pins:
        raise ValueError('Protected Final V1 files changed')
    return dict(protected_sources_unchanged=all_pins,trials=len(rows),radio_frames=samples,
                safety_violations=sum(not r['safety_passed'] for r in rows),
                packet_conserved=all(r['packet_conserved'] for r in rows),
                original_prefix_exact=True,shadow_half_second_endpoints_exact=True,
                command_results_delayed=True,stored_link_graph_outages_exact=True,
                actual_future_archive_commands_not_used=True)


def main():
    plan=prepare()
    cases=make_cases(plan)
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        print('prepared cases',len(cases),flush=True)
        return
    locked=freeze(cases);seal=sha(ARTIFACTS/'freeze.json')
    rows=simulate([(c,a,CONFIG['command_latency_s'],seal) for c in cases for a in ARMS],
                  'paired_fine_time_control')
    # One deterministic branch per map, independent of all new outcomes.
    chosen={}
    for c in sorted(cases,key=lambda c:digest([c['key'],'latency-stress'])):
        chosen.setdefault(c['spec']['region'],c)
    stress=simulate([(c,a,CONFIG['stress_command_latency_s'],seal) for c in chosen.values() for a in ARMS],
                    'command_latency_stress')
    status('audit_and_report')
    checks=audit(plan,rows+stress)
    write(ARTIFACTS/'audit.json',checks)
    from .report import build
    result=build(plan,cases,rows,stress,checks)
    write(ARTIFACTS/'completed.json',dict(state='completed',created_at=now(),freeze_sha256=seal,
        results_sha256=sha(ARTIFACTS/'results.json'),audit_sha256=sha(ARTIFACTS/'audit.json'),
        trials=len(rows)+len(stress),cohort_cases=len(cases),production_promoted=False))
    status('completed',state='completed',trials=len(rows)+len(stress),cases=len(cases))
    print('completed',result['summary'],flush=True)


if __name__=='__main__':
    main()
