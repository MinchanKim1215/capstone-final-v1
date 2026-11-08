"""Future truth is accepted only by offline targets, never by actor/selector APIs."""
import numpy as np


def timing_labels(links, endpoints, n_search):
    up = np.asarray(links, bool)
    ends = np.asarray(endpoints, int)
    if up.ndim != 3 or up.shape[1] != up.shape[2] or not 1 <= n_search < up.shape[1]:
        raise ValueError('Square RF links and a valid role split are required')
    n = up.shape[1]
    if np.any(ends < 1) or np.any(ends > len(up)):
        raise ValueError('History endpoints outside the collected RF trajectory')
    start = np.zeros_like(up)
    start[1:-1] = up[:-2] & ~up[1:-1] & ~up[2:]
    y = np.full((len(ends), n-n_search, n), 16, np.int64)
    valid = np.zeros_like(y, bool)
    for t,end in enumerate(ends):
        # The last bin requires one extra frame to confirm >=1 s persistence.
        if end+16 >= len(up):
            continue
        valid[t] = up[end-1, n_search:]
        valid[t,np.arange(n-n_search),np.arange(n_search,n)] = False
        future = start[end:end+16, n_search:]
        y[t] = np.where(future.any(0), future.argmax(0), 16)
    return y, valid


def probabilities(timing):
    """Aggregate the 17-class distribution into four explicit NEXT-event targets."""
    import torch
    p = torch.softmax(timing, -1)
    return torch.stack([p[..., :10].sum(-1), p[..., 6:8].sum(-1),
                        p[..., 6:14].sum(-1), p[..., :14].sum(-1)], -1)


def binary_targets(y):
    y = np.asarray(y)
    return np.stack([y < 10, (y >= 6)&(y < 8), (y >= 6)&(y < 14), y < 14], -1)


TARGET_NAMES = ('next_onset_lt5s', 'next_onset_3to4s', 'next_onset_3to7s', 'next_onset_lt7s')
# Unlike v5's inclusive label boundary and the first pilot's <= upper endpoints,
# v6 uses half-open [0,5), [3,4), [3,7), [0,7) consistently. The warm-start
# transfers the 0.5-second timing bins, not the pilot's aggregated metrics.
