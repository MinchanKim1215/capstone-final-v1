"""Features from published navigation and already received radio observations."""
from collections import deque

import numpy as np

from adaptive_joint_expanded.features import THRESHOLD
from final_v1_fast.controller import TIMES, WEIGHTS, runs

EARLY = (TIMES >= 3.) & (TIMES < 7.)
SHORT = (TIMES >= .5) & (TIMES < 3.)
FEATURE_NAMES = tuple(f'margin_at_{t:.1f}s' for t in TIMES) + (
    'early_min', 'early_deficit', 'short_min', 'short_deficit', 'first_projected_cut',
    'local_margin_min', 'local_margin_mean', 'local_margin_max', 'local_link_count',
    'local_noise_dbm', 'rf_age', 'measured_receiver_fraction', 'search_progress', 'sharing_phase',
    'drop_max_10s', 'drop_max_30s', 'large_drop_count_30s', 'last_large_drop_age',
    'local_level_change_5s', 'local_peak_minus_current_30s', 'recent_reports_30s',
    'local_drop_recent',
)
N_FEATURES = len(FEATURE_NAMES)
N_CANDIDATE = 11
MAX_OPTIONS = 9


class RadioHistory:
    def __init__(self, nodes):
        self.stamps = np.full(nodes, -1e9)
        self.margins = np.full(nodes, np.nan)
        self.samples = deque()

    def accept(self, now, sinr, stamps):
        margin = np.asarray(sinr) - THRESHOLD
        updated = (stamps > self.stamps + 1e-8) & (stamps <= now + 1e-8) & np.isfinite(margin)
        comparable = updated & np.isfinite(self.margins) & (stamps - self.stamps <= 2.1)
        drop = float(np.max(np.r_[0., self.margins[comparable] - margin[comparable]]))
        if updated.any():
            self.stamps[updated] = stamps[updated]
            self.margins[updated] = margin[updated]
            self.samples.append((now, float(np.mean(margin[updated])), drop))
        while self.samples and self.samples[0][0] < now - 30.:
            self.samples.popleft()

    def values(self, now, local_mean):
        recent = [item for item in self.samples if item[0] >= now - 10.]
        large = [item for item in self.samples if item[2] >= 6.]
        older = [item[1] for item in recent if item[0] <= now - 4.]
        level_change = local_mean - older[-1] if older else 0.
        peak = max([local_mean] + [item[1] for item in self.samples])
        return (max([0.] + [item[2] for item in recent]), max([0.] + [item[2] for item in self.samples]),
                len(large), now - large[-1][0] if large else 31., level_change,
                peak - local_mean, len(self.samples))


def deficit(margin, mask=None):
    loss = np.maximum(0., 1.5 - margin) * WEIGHTS
    return float(loss.sum() if mask is None else loss[mask].sum())


def physical_risk(margin):
    return deficit(margin, EARLY) + .25 * max(0., -float(np.min(margin[EARLY])))


def extract(margin, estimate, ns, history, now, progress, sharing):
    observed = estimate.power[:, ns] - estimate.noise[ns] - THRESHOLD
    observed = observed[np.isfinite(observed)]
    local_mean = float(observed.mean()) if len(observed) else -20.
    cuts = runs(margin)
    raw = [*margin, float(margin[EARLY].min()), deficit(margin, EARLY),
           float(margin[SHORT].min()), deficit(margin, SHORT), cuts[0] if cuts else 9.,
           float(observed.min()) if len(observed) else -20., local_mean,
           float(observed.max()) if len(observed) else -20., len(observed), float(estimate.noise[ns]),
           min(31., now - estimate.latest_stamp_s), float(estimate.measured.mean()), progress, float(sharing),
           *history.values(now, local_mean), float(estimate.drop_db)]
    output = np.clip(np.asarray(raw, np.float32), -150., 150.)
    if output.shape != (N_FEATURES,) or not np.isfinite(output).all():
        raise ValueError('Invalid causal feature vector')
    return output
