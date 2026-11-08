"""Register a finer high-score threshold comparison using development data only."""
import numpy as np

from . import ROOT,ARTIFACTS
from .protocol import CAP,now,read,sha,write
from .training import FAMILIES,load,metrics,predictions,restore,window


def main():
    if (ARTIFACTS/"results.json").exists():
        raise ValueError("Register threshold refinement before confirmation results are opened")
    freeze = read(ARTIFACTS/"freeze.json")
    records = read(ARTIFACTS/"development_manifest.json")["records"]
    calibration = [load(r) for r in records if read(r["path"])["contract"]["task"]["scope"]=="calibration"]
    selection = [load(r) for r in records if read(r["path"])["contract"]["task"]["scope"]=="selection"]
    choices = {}
    for family in (*FAMILIES,"legacy","drop_rule"):
        choice = freeze["choices"][family]
        model = restore(freeze["models"][family]) if family in FAMILIES else None
        p,q = (predictions(f,family,model,choice.get("rounds")) for f in (calibration,selection))
        finite = np.concatenate([s[np.isfinite(s)] for s in p])
        quantiles = np.unique(np.r_[np.linspace(.4,.98,60),np.linspace(.98,.9999,150),1.])
        points = np.unique(np.r_[float(finite.max())+.01,np.quantile(finite,quantiles)])
        options = []
        for threshold in points:
            cal = metrics(calibration,p,threshold,window(family))
            if cal["false_per_hour"]<=CAP:
                options.append((float(threshold),cal))
        threshold,cal = max(options,key=lambda item:(item[1]["recall"],item[1]["precision"],
                                                    -item[1]["false_per_hour"],item[0]))
        choices[family] = dict(rounds=choice.get("rounds"),threshold=threshold,calibration=cal,
                              selection=metrics(selection,q,threshold,window(family),True))
    output = dict(registered_at=now(),before_confirmation_metrics_opened=True,
                  reason="The primary grid has sparse coverage above the 99.4th percentile. "
                         "Refine upper-tail thresholds at fixed model iterations for a separate comparison.",
                  freeze_sha256=sha(ARTIFACTS/"freeze.json"),source_sha256=sha(ROOT/"refine.py"),
                  development_manifest_sha256=sha(ARTIFACTS/"development_manifest.json"),choices=choices)
    write(ARTIFACTS/"threshold_refinement.json",output)
    print("Refined thresholds registered before confirmation metrics",flush=True)
    for family,choice in choices.items():
        print(family,choice["threshold"],choice["selection"]["recall"],choice["selection"]["false_per_hour"],flush=True)


if __name__=="__main__":
    main()
