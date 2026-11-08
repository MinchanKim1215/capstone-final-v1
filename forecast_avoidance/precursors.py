"""Audit what the first relay actually received before attributed V17 cuts."""
from collections import Counter, defaultdict

from . import ARTIFACTS
from .protocol import V17, read, sha, write


def audit():
    results = defaultdict(Counter)
    manifest = read(V17 / 'survey_manifest.json')
    for source in ('archived', 'historical'):
        for record in manifest[source]:
            if sha(record['path']) != record['sha256']:
                raise ValueError('Cause evidence changed')
            data = read(record['path'])
            if source == 'archived':
                if sha(record['patterns_path']) != record['patterns_sha256']:
                    raise ValueError('Observation evidence changed')
                cuts = data['network_events']
                patterns = read(record['patterns_path'])['rows']
            else:
                cuts = data['cause_analysis']['network_events']
                patterns = data['observed_precursors']
            samples = {(row['onset_s'], row['lead_s']): row for row in patterns}
            for cut in cuts:
                single = {r[0] for r in cut['first_second_minimal_repairs'] if len(r)==1}
                if 'interference' in single:
                    group = 'interference_evidence'
                elif 'building' in single:
                    group = 'building_without_interference_evidence'
                else:
                    continue
                row = results[group]
                row['events'] += 1
                before = samples.get((cut['onset_s'], 3))
                prior = samples.get((cut['onset_s'], 6))
                if before is None:
                    continue
                pre = before['pre_cut']
                normal = before['comparison']
                noise = pre['local_noise_db']
                previous = prior['pre_cut']['local_noise_db'] if prior is not None else None
                if noise is not None and previous is not None:
                    row['noise_pairs_3s_6s'] += 1
                    row['local_noise_high_and_rising'] += noise >= -103. and (noise-previous)/3. >= .4
                if normal is not None and noise is not None and normal['local_noise_db'] is not None:
                    row['matched_noise'] += 1
                    row['high_noise_before'] += noise >= -103.
                    row['high_noise_normal'] += normal['local_noise_db'] >= -103.
                if normal is not None and pre['minimum_received_margin_db'] is not None and normal['minimum_received_margin_db'] is not None:
                    row['matched_margin'] += 1
                    row['weak_before'] += pre['weak_peer_count'] > 0
                    row['weak_normal'] += normal['weak_peer_count'] > 0
    result = dict(groups=results, local_only=True, source_manifest_sha256=sha(V17 / 'survey_manifest.json'))
    write(ARTIFACTS / 'precursor_audit.json', result)
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
