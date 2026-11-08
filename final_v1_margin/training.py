"""Matched supervised ablations with calibration before independent confirmation."""
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

from forecast_learning.control import load_profile
from forecast_learning.training import PARAMS,make_score

from . import ROOT,ARTIFACTS
from .features import CHANNELS,NAMES,maximum_drop,HALF_SECOND_COLUMNS
from .protocol import CAP,read,sha,status,write

ITERATIONS = (160,420)
FAMILIES = ("physics_early","current_early","rf_early","augmented_early","rf_imminent")
CURRENT_COLUMNS = tuple(NAMES.index(f"{channel}_{field}") for channel in CHANNELS
                        for field in ("margin_min","margin_mean","available_fraction","age_max"))


@dataclass
class Flight:
    row:dict
    x:np.ndarray
    rf:np.ndarray
    times:np.ndarray
    network:np.ndarray
    onsets:np.ndarray
    aligned:np.ndarray

    @property
    def support(self):
        return np.isfinite(self.x).any(axis=1)

    @property
    def usable(self):
        return self.support & (self.times+7.<=self.row["duration_s"])

    def label(self,window):
        low,high = window
        up = self.network[(self.times*2).astype(int)-1]
        return up & (np.searchsorted(self.onsets,self.times+high)>np.searchsorted(self.onsets,self.times+low))


def load(record):
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Margin feature metadata changed")
    row = read(record["path"])
    if sha(row["archive_path"]) != row["archive_sha256"]:
        raise ValueError("Margin input feature archive changed")
    with np.load(row["archive_path"],allow_pickle=False) as z:
        return Flight(row,z["x"],z["rf"],z["times"],z["network"],z["onsets"],z["event_aligned"])


def window(name):
    return (.5,3.) if name=="rf_imminent" else (3.,7.)


def inputs(flight,family):
    if family.startswith("physics"):
        return flight.x
    if family.startswith("rf"):
        return flight.rf
    if family=="current_early":
        return np.concatenate((flight.x,flight.rf[:,CURRENT_COLUMNS]),axis=-1)
    return np.concatenate((flight.x,flight.rf),axis=-1)


def predictions(flights,family,model=None,rounds=None):
    result = []
    _,profile,old,_ = load_profile()
    for f in flights:
        score = np.full(len(f.times),np.nan)
        keep = f.support
        if family=="legacy":
            from forecast_learning.features import COLUMNS
            physical = f.x[:,COLUMNS.index("held_shield")]
            score[keep] = make_score(old,profile,f.x[keep],physical[keep])
        elif family=="drop_rule":
            score = maximum_drop(f.rf)
            score[~keep] = np.nan
        elif family=="half_second_rule":
            score = maximum_drop(f.rf,HALF_SECOND_COLUMNS)
            score[~keep] = np.nan
        else:
            score[keep] = model.predict(inputs(f,family)[keep],num_iteration=rounds,num_threads=1)
        result.append(score)
    return result


def case(f,score,threshold,win):
    low,high = win
    active = score>=threshold
    alerts = f.times[active & ~np.r_[False,active[:-1]]]
    hit = np.zeros(len(f.onsets),dtype=bool)
    lead = []
    for tick in alerts:
        begin,end = np.searchsorted(f.onsets,[tick+low,tick+high])
        for i in range(begin,end):
            if not hit[i]:
                hit[i] = True
                lead.append(float(f.onsets[i]-tick))
                break
    return dict(hours=f.row["duration_s"]/3600.,events=len(hit),hits=int(hit.sum()),alerts=len(alerts),
                hit_mask=hit,lead=lead)


def metrics(flights,predicted,threshold,win=(3.,7.),detail=False):
    cases = [case(f,p,threshold,win) for f,p in zip(flights,predicted)]
    total = {key:sum(c[key] for c in cases) for key in ("hours","events","hits","alerts")}
    total.update(recall=total["hits"]/max(1,total["events"]),
                 precision=total["hits"]/max(1,total["alerts"]),
                 false_per_hour=(total["alerts"]-total["hits"])/total["hours"])
    if detail:
        y = np.concatenate([f.label(win)[f.usable & np.isfinite(p)] for f,p in zip(flights,predicted)])
        p = np.concatenate([p[f.usable & np.isfinite(p)] for f,p in zip(flights,predicted)])
        leads = [lead for c in cases for lead in c["lead"]]
        total.update(ap=float(average_precision_score(y,p)) if y.any() else None,
                     auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                     median_lead_s=float(np.median(leads)) if leads else None,
                     events_per_flight=[c["events"] for c in cases],
                     hits_per_flight=[c["hits"] for c in cases])
    return total


def calibrate(flights,predicted,win,cap):
    finite = np.concatenate([p[np.isfinite(p)] for p in predicted])
    points = np.unique(np.r_[float(finite.max())+.01,np.quantile(finite,np.linspace(.4,1.,101))])
    options = []
    for threshold in points:
        value = metrics(flights,predicted,float(threshold),win)
        if value["false_per_hour"] <= cap+1e-9:
            options.append((float(threshold),value))
    return max(options,key=lambda item:(item[1]["recall"],item[1]["precision"],-item[1]["false_per_hour"],item[0]))


def restore(info):
    if sha(info["path"]) != info["sha256"]:
        raise ValueError("Trained margin model changed")
    return lgb.Booster(model_str=Path(info["path"]).read_text(encoding="utf-8"))


def fit(records):
    path = ARTIFACTS/"freeze.json"
    if path.exists():
        frozen = read(path)
        if frozen["training_source_sha256"] != sha(ROOT/"training.py"):
            raise ValueError("Frozen trainer changed")
        for model in frozen["models"].values():
            restore(model)
        return frozen
    grouped = {scope:[load(r) for r in records if read(r["path"])["contract"]["task"]["scope"]==scope]
               for scope in ("train","calibration","selection")}
    base = grouped["train"]
    calibration,selection = grouped["calibration"],grouped["selection"]
    models,choices = {},{}
    for family in FAMILIES:
        status("fit_margin_predictor",current=family)
        x = np.concatenate([inputs(f,family)[f.usable] for f in base])
        y = np.concatenate([f.label(window(family))[f.usable] for f in base])
        names = list(NAMES) if family.startswith("rf") else None
        dataset = lgb.Dataset(x,label=y,feature_name=names or "auto")
        model = lgb.train(PARAMS,dataset,num_boost_round=ITERATIONS[-1])
        file = ARTIFACTS/"models"/f"{family}.txt"
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(model.model_to_string(),encoding="utf-8")
        models[family] = dict(path=str(file),sha256=sha(file),dimension=x.shape[1],
                              rows=len(x),positive=int(y.sum()))
        options = []
        for rounds in ITERATIONS:
            p,q = (predictions(f,family,model,rounds) for f in (calibration,selection))
            for cap in (3.,3.5,4.):
                threshold,cal = calibrate(calibration,p,window(family),cap)
                dev = metrics(selection,q,threshold,window(family),True)
                options.append(dict(rounds=rounds,threshold=threshold,calibration_cap=cap,calibration=cal,selection=dev))
        feasible = [o for o in options if o["selection"]["false_per_hour"]<=CAP]
        choices[family] = max(feasible or options,key=lambda o:(o["selection"]["recall"],o["selection"]["precision"],
                                                               -o["selection"]["false_per_hour"]))
        print("fit",family,len(x),"rows",x.shape[1],"features",choices[family]["rounds"],flush=True)
    for family in ("legacy","drop_rule"):
        p,q = (predictions(f,family) for f in (calibration,selection))
        threshold,cal = calibrate(calibration,p,(3.,7.),CAP)
        choices[family] = dict(threshold=threshold,calibration=cal,selection=metrics(selection,q,threshold,detail=True))
    choices["half_second_rule"] = dict(threshold=6.,selection=metrics(
        selection,predictions(selection,"half_second_rule"),6.,detail=True))
    frozen = dict(models=models,choices=choices,parameters=PARAMS,
                  training_rows=models["physics_early"]["rows"],new_feature_count=len(NAMES),
                  development_manifest_sha256=sha(ARTIFACTS/"development_manifest.json"),
                  training_source_sha256=sha(ROOT/"training.py"),
                  current_value_control_sha256=sha(ARTIFACTS/"current_value_control.json"))
    write(path,frozen)
    return frozen
