"""Validate the upgrade package before a Final V1 flight."""
import torch

from forecast_rebuild.contracts import BASELINE, read, sha

from . import ARTIFACTS, ROOT
from .collection import PLAN
from .training import FIT_FILE, NETWORK_FILE, TIMING_FILE

FREEZE = ARTIFACTS / 'freeze.json'
RESULT = ARTIFACTS / 'results.json'
INTEGRATION = ARTIFACTS / 'integration.json'


def fingerprints():
    sources = ('__init__.py', 'capture.py', 'collection.py', 'model.py', 'training.py',
               'alerts.py', 'bridge.py', 'release.py', 'campaign.py')
    return dict(protocol=sha(PLAN), baseline=sha(BASELINE), training=sha(FIT_FILE),
                timing=sha(TIMING_FILE), network=sha(NETWORK_FILE),
                source={name: sha(ROOT / name) for name in sources})


def load_frozen(expected=None):
    if expected is not None and sha(FREEZE) != expected:
        raise ValueError('Frozen upgrade manifest has changed')
    freeze = read(FREEZE)
    if freeze['fingerprints'] != fingerprints():
        raise ValueError('Frozen Transformer upgrade files have changed')
    timing = torch.load(TIMING_FILE, map_location='cpu', weights_only=True)
    network = torch.load(NETWORK_FILE, map_location='cpu', weights_only=True)
    return dict(timing=timing['timing'], network=network['network']), freeze


def installed():
    record = read(INTEGRATION)
    if sha(RESULT) != record['results_sha256']:
        raise ValueError('Final V1 upgrade evidence has changed')
    result = read(RESULT)
    if (record['default'] != result['promotion']['passed']
            or record['freeze_sha256'] != result['freeze_sha256']):
        raise ValueError('Final V1 upgrade selection disagrees with the frozen result')
    return load_frozen(record['freeze_sha256'])
