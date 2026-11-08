"""Exercise native entry points and compare unchanged control with an archive."""
import numpy as np

from . import ARTIFACTS
from .protocol import prepare, write, now
from . import diagnostics, runner
from adaptive_joint_expanded.features import OFFSETS


def main():
    plan = prepare()
    item = plan['archived_records'][0]
    source = item['source_record']
    runner.initialize()
    base, base_data = runner.run(source['spec'], 'baseline', max_steps=18)
    nav, nav_data = runner.run(source['spec'], 'navigation_control', max_steps=18)
    with np.load(source['array_path'], allow_pickle=False) as old:
        for name in ('commands', 'nodes', 'links'):
            np.testing.assert_array_equal(base_data[name], old[name][:len(base_data[name])])
            np.testing.assert_array_equal(base_data[name], nav_data[name])
    for key in ('generated_packets', 'delivered_packets', 'timely_pdr', 'retries', 'beacon_rx', 'beacon_tx'):
        if base[key] != nav[key]:
            raise ValueError('Navigation sidecar changed existing channel/packet outcomes')
    prototypes = {}
    for arm in ('reactive', 'geometry', 'switch'):
        row, data = runner.run(source['spec'], arm, max_steps=18)
        if not row['decision_counts'] or not row['audited']['passed']:
            raise ValueError('Overlay did not reach the normal control entry point')
        first_relay = row['search_drones']
        np.testing.assert_array_equal(data['commands'][:, first_relay+1:], OFFSETS[data['original_actions'][:, 1:]])
        prototypes[arm] = row['decision_counts']
    verified = diagnostics.archived(item)
    write(ARTIFACTS / 'preflight.json', dict(passed=True, time=now(), tested_control_steps=18,
          commands_match_archive=True, beacon_and_packet_outcomes_unchanged=True,
          actual_motion_safety_checked=True, other_relays_keep_v7_actions=True,
          diagnosis=verified, modes=prototypes))
    print('Native preflight passed', prototypes, flush=True)


if __name__ == '__main__':
    main()
