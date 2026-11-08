"""Causal observations before cuts, with nearby clean windows for comparison."""
import numpy as np

from . import ARTIFACTS
from .protocol import read, sha, write
from .diagnostics import archived, connected, outage_runs


def observed_frame(values, masks, frame):
    x = values[frame]
    m = masks[frame]
    recent = m[:, 28:31].all(axis=1) & (x[:, 30] <= 2)
    margin = recent & m[:, 41]
    return dict(recent_peer_reports=int(recent.sum()),
                weak_peer_count=int(np.sum(margin & (x[:, 41] < 3))),
                minimum_received_margin_db=float(x[margin, 41].min()) if margin.any() else None,
                local_noise_db=float(np.median(x[recent, 28]-x[recent, 29])) if recent.any() else None,
                local_queue=float(x[0, 11]) if m[0, 11] else None)


def compare(values, masks, healthy):
    cuts = outage_runs(healthy)
    controls = np.array(healthy, copy=True)
    controls[:40] = False
    controls[-16:] = False
    for start, stop in cuts:
        controls[max(0, start-18):min(len(controls), stop+8)] = False
    controls = np.flatnonzero(controls)
    rows = []
    for start, _ in cuts:
        for lead in (3, 6):
            frame = start-2*lead-1
            if frame < 39 or not healthy[frame]:
                continue
            window = controls[np.abs(controls-frame) <= 240]
            other = int(window[np.argmin(np.abs(window-frame))]) if len(window) else None
            rows.append(dict(onset_s=start*.5, lead_s=lead, observation_end_s=(frame+1)*.5,
                             pre_cut=observed_frame(values, masks, frame),
                             control_end_s=(other+1)*.5 if other is not None else None,
                             comparison=observed_frame(values, masks, other) if other is not None else None))
    return rows


def survey_one(item):
    cause = archived(item)
    source = item['source_record']
    path = ARTIFACTS / 'patterns' / (item['key'] + '.json')
    contract = dict(cause_sha256=cause['sha256'], raw_sha256=source['array_sha256'], code_sha256=sha(__file__))
    if path.exists():
        if read(path)['contract'] != contract:
            raise ValueError('Precursor evidence changed')
    else:
        with np.load(source['array_path'], allow_pickle=False) as raw:
            values = raw['rf_values'][:, 0]
            masks = raw['rf_masks'][:, 0]
            healthy = connected(raw['links'], source['n_search'])
        rows = compare(values, masks, healthy)
        write(path, dict(contract=contract, rows=rows))
    return dict(cause, patterns_path=str(path), patterns_sha256=sha(path))
