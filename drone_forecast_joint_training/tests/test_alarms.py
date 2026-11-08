import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from solo.alarms import AlarmMonitor
from solo.contracts import atomic_json,sha,sources,load_plan,object_hash
from solo.evaluate import analysis_sources


class AlarmTests(unittest.TestCase):
    def test_frozen_alarm_hysteresis_warmup_self_mask_and_disabled_rule(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'weights.pt';checkpoint.write_bytes(b'alarm identity')
            selection=Path(d)/'selection.json'
            atomic_json(selection,dict(format='standalone-transformer-selection-v5',validation_cases=len(load_plan()['validation']),
                selected_sha256=sha(checkpoint),source_sha256=sources(),plan_sha256=object_hash(load_plan()),analysis_sha256=analysis_sources(),
                alarm_policy={'5s':dict(enabled=True,head=1,threshold=.7,exit_threshold=.3),
                              '8s':dict(enabled=False,head=2,threshold=.7,exit_threshold=.3)}))
            m=AlarmMonitor(checkpoint,selection);p=np.full((5,8,4),.9);s=np.ones((5,8),bool)
            self.assertEqual(m.update(18,p,s),{'5s':[],'8s':[]})
            out=m.update(21,p,s);self.assertEqual(len(out['5s']),35);self.assertFalse(out['8s'])
            p[:]=.5;self.assertFalse(m.update(24,p,s)['5s'])
            p[:]=.2;self.assertFalse(m.update(27,p,s)['5s'])
            p[:]=.9;self.assertEqual(len(m.update(30,p,s)['5s']),35)
            with self.assertRaisesRegex(ValueError,'Consecutive'):m.update(36,p,s)

    def test_reduced_validation_cannot_activate_alarms(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pilot.json';atomic_json(p,dict(format='standalone-transformer-selection-v5',validation_cases=8))
            with self.assertRaisesRegex(ValueError,'full-validation'):AlarmMonitor('not-opened.pt',p)
