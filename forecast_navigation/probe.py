"""Train/selection-only gate for replacing the tentative path generator."""
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
import forecast_graph
from forecast_graph.contracts import read,write,sha,now,source_records
from forecast_graph.trace import attach
from .preview import capture,preview


def evaluate(record):
    from adaptive_joint_expanded.world import World
    env=World(**record['spec'],deployment=record['deployment'],observer=None)
    telemetry=attach(env);original=telemetry.issue;plans=[];prior=[]
    def issued(time_s,desired):
        original(time_s,desired);old=np.array(telemetry.planned_nodes)
        state=capture(env.stream)
        proposed=preview(env.city,env.design,env.scenario,state,env.baseline_duration,env.k)
        new=np.concatenate([old[:1],proposed],axis=0)
        prior.append(old);plans.append(new)
        #The probe logs both hypothetical plans; never changes motion or RF.
    telemetry.issue=issued
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];positions=a['positions'];nodes=a['nodes']
    for k,c in enumerate(commands):
        env.step(c);np.testing.assert_equal(env.nodes,nodes[k+1])
    errors={name:[[] for _ in range(9)] for name in ('original','mission')}
    for k,(old,new) in enumerate(zip(prior,plans)):
        for h in range(9):
            absolute=k*3+h;lo=int(absolute//3);f=(absolute-lo*3)/3
            if lo+1>=len(positions):continue
            actual=positions[lo]*(1-f)+positions[lo+1]*f
            for name,path in [('original',old),('mission',new)]:
                a=min(int(h//3),len(path)-2);z=(h-a*3)/3;predicted=path[a]*(1-z)+path[a+1]*z
                errors[name][h].append(np.linalg.norm(predicted-actual,axis=-1))
    summary={name:[float(np.sqrt(np.mean(np.concatenate(p)**2))) for p in parts] for name,parts in errors.items()}
    value=dict(key=record['key'],role=record['role'],region=record['region'],rmse_by_horizon=summary,
        no_RF_future_input=True,trajectory_unchanged=True,source_sha256=record['array_sha256'],created_at=now())
    write(ARTIFACTS/(record['key']+'.json'),value);return value


if __name__=='__main__':
    #Predeclared research gate; no V11 confirmation outcomes are consulted.
    plan=dict(created_at=now(),scope='Train pilot then selection45 only',criterion='At least30% lower aggregate3-to7s positionRMSE before any extra fitting',
        inputs='Current navigation planner memory and public map only; held current relay correction hypothesis',
        original_V11_freeze_sha256=sha(forecast_graph.ARTIFACTS/'freeze.json'),fit_permitted_only_after_gate=True)
    if not (ARTIFACTS/'plan.json').exists():write(ARTIFACTS/'plan.json',plan)
    result=evaluate(source_records()[0]);print(result)
