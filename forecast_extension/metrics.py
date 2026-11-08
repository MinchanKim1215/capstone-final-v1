"""Exact preserved first-alert rule, vectorized for operating-point sweeps."""
import numpy as np
from forecast_rebuild.targets import onsets
from forecast_rebuild.metrics import binary,combine_alerts


def alert_counts(links,ends,score,support,n_search,threshold=.5):
    times=np.asarray(ends)*.5;end=len(links)*.5-8.5
    use=np.flatnonzero((times>=21)&(times<=end));nr,n=score.shape[1:]
    # The preserved implementation compares against a float64 np.where
    # threshold array. Explicit float64 prevents NumPy weak-scalar promotion
    # from rounding a quantile threshold to float32 at an exact boundary.
    p=np.asarray(score,dtype=np.float64)[use];known=np.asarray(support)[use]&np.isfinite(p)
    enter=known&(p>=threshold);leave=~known|(p<threshold*.5)
    ordinal=np.arange(len(use))[:,None,None]
    last_enter=np.maximum.accumulate(np.where(enter,ordinal,-1),axis=0)
    last_leave=np.maximum.accumulate(np.where(leave,ordinal,-1),axis=0)
    active=last_enter>last_leave;starts=active.copy()
    if len(use)>1:starts[1:] &= ~active[:-1]
    events=onsets(links);first=correct=hits=total=0
    for r in range(nr):
        for j in range(n):
            if j==r+n_search:continue
            es=np.flatnonzero(events[:,r+n_search,j])*.5;alerts=times[use][starts[:,r,j]]
            first+=len(alerts)
            correct+=int((np.searchsorted(es,alerts+7,side='left')>np.searchsorted(es,alerts+3,side='left')).sum())
            eligible=es[(es>=24)&(es<=end)];total+=len(eligible)
            hits+=int((np.searchsorted(alerts,eligible-3,side='right')>np.searchsorted(alerts,eligible-7,side='right')).sum())
    return dict(first_alerts=int(first),timely_correct_first_alerts=int(correct),eligible_directed_link_events=int(total),
                event_hits_3s=int(hits),drone_hours=max(0.,end-21)*nr/3600)

