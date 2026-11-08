"""전수 또는 랜덤 설계 탐색. 같은 환경·seed를 모든 설계에 적용한다."""
# 초기 짧은 시험용 범용 탐색 코드입니다. 현재 전체 수색 10,240회는 루트 full_study.py를 사용합니다.
# 이 파일에는 랜덤 표본·민감도 설정을 지원하는 코드가 남아 있지만 현재 비교 화면의 실험 묶음은 두 개입니다.
# 과거 120초 시험과 현재 임무 완료까지의 결과를 합산하지 않습니다.
from dataclasses import asdict, replace
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
import numpy as np
from .config import Design,Scenario,FORMATIONS,from_dict
from .city import City
from .environment import SwarmEnvironment
from .metrics import pareto_mask

ROOT=Path(__file__).resolve().parents[1]
_CITIES={}


# 초기 엔진 소스 해시를 기록합니다. 한글 주석만 바뀌어도 파일 해시는 달라질 수 있습니다.
def core_hashes():
    names=("config","city","motion","radio","network","metrics","environment")
    return {name:hashlib.sha256((ROOT/"swarm_sim"/(name+".py")).read_bytes()).hexdigest() for name in names}


# 임시 파일 작성 후 교체해 저장 중단으로 인한 반쪽 JSON을 줄입니다.
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temp,path)


# 한글 Excel용 UTF-8 BOM CSV를 임시 파일에 완성한 뒤 교체합니다.
def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8-sig");return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    os.replace(temp,path)


# 설정 내용을 고정 순서 JSON으로 바꿔 초기 실험 식별자를 만듭니다.
def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()[:16]


# 초기 범용 탐색의 전수 또는 중복 없는 랜덤 설계 선택입니다. 현재512개 전수는 full_study.designs를 봅니다.
def design_choices(config,mode="exhaustive",samples=50,seed=20261009):
    if mode=="random" and samples<1:raise ValueError("랜덤 설계 수는 1 이상이어야 합니다.")
    axes=config.get("design_axes",{})
    ns=axes.get("n_drones",list(range(6,11)))
    forms=axes.get("formation",list(FORMATIONS));spaces=axes.get("spacing_m",[20,40,60])
    scopes=axes.get("formation_scope",["search"]);layouts=axes.get("relay_layout",["backbone"])
    decision=config.get("decision_scenario_axes",{})
    keys=list(decision);variants=list(product(*(decision[k] for k in keys))) or [()]
    all_choices=[];seen=set()
    for n,f,space,scope,layout in product(ns,forms,spaces,scopes,layouts):
        allowed=range(n-1) if axes.get("n_relays","all")=="all" else axes["n_relays"]
        for nr in allowed:
            if nr>n-2:continue
            # 동일 물리 설계가 되는 무중계·전체대형의 중계 배치 축은 중복 제거한다.
            effective_layout=layout if nr and scope=="search" else "backbone"
            d=Design(int(n),int(nr),f,float(space),scope,effective_layout).validate()
            for variant in variants:
                decision_change=dict(zip(keys,variant));item={"design":asdict(d),"decision":decision_change}
                ident=digest(item)
                if ident not in seen:all_choices.append(item);seen.add(ident)
    if mode=="exhaustive":return all_choices
    rng=np.random.default_rng(seed)
    continuous=config.get("random_spacing_range_m")
    if continuous:
        low,high=map(float,continuous)
        if low<10 or high<=low:raise ValueError("랜덤 간격 범위가 유효하지 않습니다.")
        chosen=[];seen=set()
        for _ in range(samples*100+100):
            n=int(rng.choice(ns));allowed=list(range(n-1)) if axes.get("n_relays","all")=="all" else [r for r in axes["n_relays"] if r<=n-2]
            nr=int(rng.choice(allowed));scope=str(rng.choice(scopes));layout=str(rng.choice(layouts)) if nr and scope=="search" else "backbone"
            d=Design(n,nr,str(rng.choice(forms)),round(float(rng.uniform(low,high)),6),scope,layout).validate()
            dec={}
            for key in keys:
                value=rng.choice(decision[key]);dec[key]=value.item() if isinstance(value,np.generic) else value
            item={"design":asdict(d),"decision":dec};ident=digest(item)
            if ident not in seen:chosen.append(item);seen.add(ident)
            if len(chosen)==samples:return chosen
        raise ValueError("지정한 랜덤 설계 수를 생성할 수 없습니다.")
    if samples>len(all_choices):raise ValueError("랜덤 표본 수가 서로 다른 이산 설계 수보다 많습니다.")
    chosen=[];seen=set()
    # N을 균등하게, 그 N에서 가능한 중계 수를 균등하게 뽑는다.
    while len(chosen)<samples:
        n=int(rng.choice(ns));subset=[x for x in all_choices if x["design"]["n_drones"]==n]
        item=subset[int(rng.integers(len(subset)))];ident=digest(item)
        if ident not in seen:chosen.append(item);seen.add(ident)
    return chosen


# 설계·지역·환경 축·난수를 곱해 초기 시험 작업을 만듭니다. 같은 키를 설계와 환경 축에 중복 지정하지 않습니다.
def episode_tasks(config,choices):
    base=from_dict(Scenario,config.get("scenario",{}))
    env_axes=config.get("environment_axes",{});keys=list(env_axes)
    envs=[dict(zip(keys,x)) for x in product(*(env_axes[k] for k in keys))] or [{}]
    seeds=config.get("seeds",[1,2,3]);regions=config.get("regions",["teheran"])
    tasks=[]
    for item,region,env,seed in product(choices,regions,envs,seeds):
        overlaps=set(env)&set(item["decision"])
        if overlaps:raise ValueError(f"설계/환경 축이 겹칩니다: {overlaps}")
        scenario=replace(base,**item["decision"],**env).validate()
        ident=digest({"choice":item,"region":region,"scenario":asdict(scenario),"seed":int(seed)})
        tasks.append({"id":ident,"design_id":digest(item),"design":item["design"],
                      "decision":item["decision"],"region":region,"scenario":asdict(scenario),
                      "environment":env,"seed":int(seed)})
    return tasks


# 초기 고정 시간 시험 한 건입니다. 잘못된 배치는 invalid로 남기며 다른 설계로 몰래 교체하지 않습니다.
def execute(task):
    d=from_dict(Design,task["design"]);s=from_dict(Scenario,task["scenario"])
    geometry=(task["region"],s.altitude_m,s.horizontal_clearance_m,s.vertical_clearance_m,
              s.grid_m,s.sensor_radius_m,s.sensor_occlusion,s.height_case,s.sweep_spacing_m,s.direction)
    try:
        city=_CITIES.get(geometry)
        if city is None:
            if len(_CITIES)>=8:_CITIES.pop(next(iter(_CITIES)))
            city=City(ROOT/"maps"/(task["region"]+".scene.local.json"),s);_CITIES[geometry]=city
        r=SwarmEnvironment(city.file_path,d,s,task["seed"],city=city).run()
        return {**task,"status":"ok","map":r["map"],"jammer_xyz":r["jammer_xyz"],"metrics":r["metrics"]}
    except ValueError as exc:
        # 폭 초과 등 유효하지 않은 조건은 기록하며 다른 설계로 몰래 바꾸지 않는다.
        return {**task,"status":"invalid","reason":str(exc)}


# 실행별 설정과 지표를 CSV의 한 행으로 펼칩니다.
def flatten(r):
    row={"episode_id":r["id"],"design_id":r["design_id"],"region":r["region"],"seed":r["seed"],
         "status":r["status"],**r["design"],"relay_fraction":r["design"]["n_relays"]/r["design"]["n_drones"],
         "n_search":r["design"]["n_drones"]-r["design"]["n_relays"]}
    row.update({"decision_"+k:v for k,v in r["decision"].items()})
    row.update({"env_"+k:v for k,v in r["environment"].items()})
    if r["status"]=="ok":row.update(r["metrics"])
    else:row["invalid_reason"]=r["reason"]
    return row


# 초기 시험의 설계별 평균·최저·최고·표준편차와5축파레토를 만듭니다. 현재 전체 결과의6축과 구분합니다.
def summaries(records):
    groups={}
    for r in records:
        if r["status"]=="ok":groups.setdefault(r["design_id"],[]).append(r)
    out=[]
    for ident,items in groups.items():
        row={"design_id":ident,**items[0]["design"],"n_search":items[0]["design"]["n_drones"]-items[0]["design"]["n_relays"],
             "relay_fraction":items[0]["design"]["n_relays"]/items[0]["design"]["n_drones"],
             **{"decision_"+k:v for k,v in items[0]["decision"].items()},"episodes":len(items),"regions":len({x["region"] for x in items})}
        for key in items[0]["metrics"]:
            a=np.array([x["metrics"][key] for x in items if x["metrics"][key] is not None],float)
            row[key+"_mean"]=float(a.mean()) if len(a) else None
            row[key+"_min"]=float(a.min()) if len(a) else None
            row[key+"_max"]=float(a.max()) if len(a) else None
            row[key+"_sd"]=float(a.std(ddof=1)) if len(a)>1 else None
        out.append(row)
    if out:
        objectives=["search_connected_mean_mean","pdr_mean","shared_coverage_final_mean","total_travel_m_mean","n_drones"]
        mask=pareto_mask([[r[k] for k in objectives] for r in out],[True,True,True,False,False])
        for r,m in zip(out,mask):r["pareto_on_declared_5_objectives"]=bool(m)
    return out


# 초기 범용 탐색을 실행·저장·재개합니다. 설정·지도·코드 서명이 다르면 기존 폴더에 이어 쓰지 못하게 합니다.
def run_study(config,out,mode="exhaustive",samples=50,seed=20261009,jobs=1,max_runs=None,resume=False,dry_run=False):
    if jobs<1 or (max_runs is not None and max_runs<1):raise ValueError("병렬 작업 수와 실행 제한은 1 이상이어야 합니다.")
    choices=design_choices(config,mode,samples,seed);tasks=episode_tasks(config,choices)
    all_count=len(tasks)
    if max_runs is not None:tasks=tasks[:max_runs]
    maps={r:hashlib.sha256((ROOT/"maps"/(r+".scene.local.json")).read_bytes()).hexdigest() for r in config.get("regions",["teheran"])}
    manifest={"config":config,"mode":mode,"design_seed":seed,"candidate_designs":len(choices),
              "planned_episodes":all_count,"requested_episodes":len(tasks),"partial_by_max_runs":len(tasks)<all_count,
              "map_sha256":maps,"core_code_sha256":core_hashes(),
              "signature":digest({"config":config,"mode":mode,"samples":samples if mode=="random" else None,"seed":seed,"maps":maps,"code":core_hashes()}),
              "contains_prediction":False,"contains_risk_based_repositioning":False,"composite_score":False,
              "routing_information":"ideal_periodic_global_snapshot","duration":"coarse_search_screen_if_shorter_than_full_mission"}
    if dry_run:return manifest
    out=Path(out);out.mkdir(parents=True,exist_ok=True);episode_dir=out/"episodes";episode_dir.mkdir(exist_ok=True)
    if (out/"manifest.json").exists():
        old=json.loads((out/"manifest.json").read_text())
        if not resume:raise ValueError("이미 결과가 있는 폴더입니다. 새 경로 또는 --resume을 사용하세요.")
        if old["signature"]!=manifest["signature"]:raise ValueError("지도·설정이 다른 결과에 이어 쓸 수 없습니다.")
    atomic_json(out/"manifest.json",manifest)
    records=[];pending=[]
    for task in tasks:
        p=episode_dir/(task["id"]+".json")
        if resume and p.exists():records.append(json.loads(p.read_text()))
        else:pending.append(task)
    pool=ProcessPoolExecutor(max_workers=jobs) if jobs>1 else None
    stream=pool.map(execute,pending,chunksize=5) if pool else map(execute,pending)
    try:
        for result in stream:
            atomic_json(episode_dir/(result["id"]+".json"),result);records.append(result)
            if len(records)%50==0 or len(records)==len(tasks):
                print(f"완료 {len(records)}/{len(tasks)} (유효 {sum(r['status']=='ok' for r in records)})",flush=True)
    finally:
        if pool:pool.shutdown(wait=True)
    write_csv(out/"episodes.csv",map(flatten,records))
    summary=summaries(records);write_csv(out/"design_summary.csv",summary)
    write_csv(out/"pareto_candidates.csv",[r for r in summary if r["pareto_on_declared_5_objectives"]])
    atomic_json(out/"design_choices.json",choices)
    manifest.update({"completed_episodes":len(records),"valid_episodes":sum(r["status"]=="ok" for r in records),
                     "invalid_episodes":sum(r["status"]!="ok" for r in records),
                     "pareto_designs":sum(r["pareto_on_declared_5_objectives"] for r in summary)})
    atomic_json(out/"manifest.json",manifest);return manifest
