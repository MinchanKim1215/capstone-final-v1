"""Causal differences of actually received beacon and ACK measurements."""
import numpy as np

from adaptive_joint_expanded.features import THRESHOLD

LAGS = (.5,1.,2.)
CHANNELS = ("beacon_inbound","beacon_bidirectional","ack_inbound")
NAMES = tuple(
    name
    for channel in CHANNELS
    for name in (
        *(f"{channel}_{field}" for field in ("margin_min","margin_mean","available_fraction","age_max")),
        *(f"{channel}_{lag:g}s_{field}" for lag in LAGS
          for field in ("max_drop","mean_delta","min_rate","distinct_updates","rise_max")),
        *(f"{channel}_{field}" for field in ("acceleration_drop_max","acceleration_mean","acceleration_support"))
    )
)


def summarize(values, valid, operation):
    count = valid.sum(axis=1)
    if operation == "mean":
        total = np.where(valid,values,0.).sum(axis=1)
        return np.divide(total,count,out=np.full(len(values),np.nan),where=count>0)
    fill = np.inf if operation == "min" else -np.inf
    result = np.where(valid,values,fill)
    reduced = result.min(axis=1) if operation == "min" else result.max(axis=1)
    return np.where(count>0,reduced,np.nan)


def observed_features(values, masks, observer):
    """Return features at each completed half-second, without future access."""
    x = np.asarray(values,dtype=float)
    m = np.asarray(masks,dtype=bool)
    if x.ndim != 3 or x.shape != m.shape or x.shape[-1] != 48:
        raise ValueError("Expected local [frames,peers,48] telemetry")
    frames,n,_ = x.shape
    now = (np.arange(frames)+1)*.5
    peer = np.arange(n) != observer
    inbound = m[:,:,29] & m[:,:,30] & (x[:,:,30]>=0) & (x[:,:,30]<=2)
    bi = m[:,:,41] & m[:,:,30] & m[:,:,32] & m[:,:,46] & (x[:,:,46]>.5)
    bi &= (x[:,:,30]>=0) & (x[:,:,30]<=2) & (x[:,:,32]>=0) & (x[:,:,32]<=2)
    ack = m[:,:,9] & m[:,:,10] & (x[:,:,10]>=0) & (x[:,:,10]<=2)
    parts = [
        (x[:,:,29]-THRESHOLD,x[:,:,30],inbound),
        (x[:,:,41],np.maximum(x[:,:,30],x[:,:,32]),bi),
        (x[:,:,9]-THRESHOLD,x[:,:,10],ack),
    ]
    blocks = []
    for margin,age,valid in parts:
        valid = valid & peer
        stamp = now[:,None]-age
        output = [summarize(margin,valid,"min"),summarize(margin,valid,"mean"),
                  valid.sum(axis=1)/max(1,n-1),summarize(age,valid,"max")]
        deltas, supports = {}, {}
        for lag in LAGS:
            shift = int(lag*2)
            difference = np.full_like(margin,np.nan)
            elapsed = np.full_like(margin,np.nan)
            support = np.zeros_like(valid)
            if frames > shift:
                # Held copies of the same observation do not count as new samples.
                dt = stamp[shift:]-stamp[:-shift]
                support[shift:] = valid[shift:] & valid[:-shift] & (dt>1e-4)
                difference[shift:] = margin[shift:]-margin[:-shift]
                elapsed[shift:] = dt
            safe = np.where(support,elapsed,1.)
            rate = difference/safe
            output.extend([
                np.maximum(0.,-summarize(difference,support,"min")),
                summarize(difference,support,"mean"),
                summarize(rate,support,"min"),
                support.sum(axis=1)/max(1,n-1),
                np.maximum(0.,summarize(difference,support,"max")),
            ])
            deltas[lag],supports[lag] = rate,support
        # Change of 1-second slope over another 1-second interval.
        acceleration = np.full_like(margin,np.nan)
        supported = np.zeros_like(valid)
        if frames > 2:
            acceleration[2:] = deltas[1.][2:]-deltas[1.][:-2]
            supported[2:] = supports[1.][2:] & supports[1.][:-2]
        output.extend([np.maximum(0.,-summarize(acceleration,supported,"min")),
                       summarize(acceleration,supported,"mean"),supported.sum(axis=1)/max(1,n-1)])
        blocks.extend(output)
    result = np.asarray(blocks,dtype=np.float32).T
    if result.shape != (frames,len(NAMES)) or np.isinf(result).any():
        raise ValueError("Observed-margin feature contract changed")
    return result


DROP_COLUMNS = tuple(NAMES.index(f"{channel}_{lag:g}s_max_drop") for channel in CHANNELS for lag in LAGS)
HALF_SECOND_COLUMNS = tuple(NAMES.index(f"{channel}_0.5s_max_drop") for channel in CHANNELS)
TWO_SECOND_COLUMNS = tuple(NAMES.index(f"{channel}_2s_max_drop") for channel in CHANNELS)


def maximum_drop(features, columns=DROP_COLUMNS):
    values = features[:,columns]
    valid = np.isfinite(values)
    return summarize(values,valid,"max")
