"""Read-only payload logging; original motion/RF/data/beacon RNG remain intact."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,verify,status,source_records
from .codec import pack,H


def attach(env):
    from adaptive_joint_expanded.telemetry import Telemetry
    class PlanTelemetry(Telemetry):
        def issue(self,time_s,desired):
            super().issue(time_s,desired)
            #These are the already issued next3s collision-resolved controller
            #segments, available BEFORE any RF interval, not recorded future motion.
            old=env.node_log[-2];new=env.node_log[-1]
            paths=[env.city.points[old],env.city.points[new]]
            current=np.array(new)
            from swarm_sim.mission import unique_goals
            goals=unique_goals(env.city,np.asarray(desired))
            for k in range(3):
                current=np.array([env.city.next_node(int(a),int(b)) for a,b in zip(current,goals)])
                paths.append(env.city.points[current])
            self.planned_nodes=np.asarray(paths,float)

        def push(self,start,end,received,sinr,attempts,successes,last_ack,queues):
            t=self.t
            outgoing=None
            if t%2==0:
                now=t*.5;elapsed=now-self.issued_time
                future=elapsed+H
                left=np.minimum((future//3).astype(int),len(self.planned_nodes)-2)
                alpha=np.clip((future-left*3)/3,0,1)
                planned=self.planned_nodes[left]*(1-alpha[:,None,None])+self.planned_nodes[left+1]*alpha[:,None,None]
                outgoing=[]
                for j in range(self.n):
                    previous=self.x[t-1,j] if t else np.zeros((self.n,48))
                    q=previous[0,11];attempt=previous[:,5].sum();ok=previous[:,6].sum()
                    outgoing.append(pack(j,self.n,self.ns,now,self.issued_time,(end[j,:2]-start[j,:2])/.5,
                        q,attempt,ok,self.desired[j],planned[:,j],self.last[j].copy(),self.rxsin[j].copy(),self.rxdb[j].copy()))
            result=super().push(start,end,received,sinr,attempts,successes,last_ack,queues)
            if outgoing is not None:
                self.plan_tx.append(np.asarray(outgoing));self.plan_rx.append(self.counts[t].astype(bool).copy())
            return result
    old=env.telemetry
    telemetry=PlanTelemetry(env.city,env.scenario,env.seed,env.n,env.ns,max_frames=old.max_frames)
    telemetry.plan_tx=[];telemetry.plan_rx=[]
    env.telemetry=telemetry
    return telemetry


def run(record,limit=None,save=True):
    from adaptive_joint_expanded.world import World
    from forecast_rebuild.dataset import Episode
    folder=ARTIFACTS/'traces';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');meta=path.with_suffix('.json')
    pins={name:sha(ROOT/name) for name in ('codec.py','trace.py')}
    if save and meta.exists():
        old=read(meta)
        if old['source_sha256']!=record['array_sha256'] or old['source_pins']!=pins or sha(path)!=old['sha256']:
            raise ValueError('Existing causal trace changed')
        return old
    if sha(record['array_path'])!=record['array_sha256']:raise ValueError('Source archive changed')
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];nodes=a['nodes']
    ep=Episode(record) if record.get('cache_path') else None
    env=World(**record['spec'],deployment=record['deployment'],observer=None)
    tel=attach(env);started=time.monotonic()
    for k,command in enumerate(commands if limit is None else commands[:limit]):
        env.step(command)
        np.testing.assert_equal(env.nodes,nodes[k+1])
        if ep is not None:
            np.testing.assert_equal(tel.x[env.t-1,env.ns:].astype(np.float16),ep.values[env.t-1])
            np.testing.assert_equal(tel.mask[env.t-1,env.ns:],ep.masks[env.t-1])
            np.testing.assert_equal(np.asarray(env.channel_available[-6:]),ep.links[k*6:k*6+6])
    arrays=dict(tx=np.asarray(tel.plan_tx),rx=np.packbits(np.asarray(tel.plan_rx),axis=-1),
        sinr=tel.sinr[:env.t].astype(np.float16),n=np.array(env.n),ns=np.array(env.ns))
    if limit is not None:return arrays
    assert env.done
    original=read(record['metadata_path'])
    for key in ('generated_packets','delivered_packets','ttl_drops','retry_drops','overflow_drops','retries','timely_pdr'):
        if env.summary[key]!=original[key]:raise ValueError('Packet replay changed: '+key)
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    value=dict(key=record['key'],role=record['role'],region=record['region'],path=str(path),sha256=sha(path),
        source_sha256=record['array_sha256'],source_pins=pins,frames=env.t,n=env.n,ns=env.ns,
        commands_and_RF_exact=True,packet_outcomes_exact=True,beacon_airtime_unchanged=True,
        wall_seconds=time.monotonic()-started,created_at=now())
    write(meta,value);return value


def main():
    verify();rows=source_records();results={}
    status('causal_plan_and_graph_replay',complete=0,total=len(rows))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(run,r):i for i,r in enumerate(rows)}
        for future in as_completed(pending):
            results[pending[future]]=future.result()
            write(ARTIFACTS/'trace_manifest.json',dict(records=[results[i] for i in sorted(results)],created_at=now()))
            status('causal_plan_and_graph_replay',complete=len(results),total=len(rows))
            print(len(results),'/',len(rows),flush=True)
    write(ARTIFACTS/'trace_status.json',dict(state='completed',total=len(rows),created_at=now()))


if __name__=='__main__':main()

