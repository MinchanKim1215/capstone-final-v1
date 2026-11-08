import unittest
import io
from pathlib import Path
import numpy as np
from core import timing_labels, causal_support, windows, next_targets, next_probabilities, event_metrics, make_model, PROJECT


class TimingContractTests(unittest.TestCase):
    def test_every_half_second_bin_and_boundaries(self):
        for slot in range(17):
            up = np.ones((160, 8, 8), bool)
            q = 60+slot
            up[q:q+2, 3, 0] = False
            result = timing_labels(up, [60])
            self.assertTrue(result['valid'][0, 0, 0])
            self.assertEqual(result['label'][0, 0, 0], min(slot, 16))
        # An onset at exactly +8 s is outside [0,8), not the final timing bin.

    def test_one_bad_frame_is_not_persistent(self):
        up = np.ones((160, 8, 8), bool)
        up[65, 3, 0] = False
        self.assertEqual(timing_labels(up, [60])['label'][0, 0, 0], 16)

    def test_first_event_not_later_repeat(self):
        up = np.ones((160, 8, 8), bool)
        up[61:63, 3, 0] = False
        up[68:70, 3, 0] = False
        row = timing_labels(up, [60])
        self.assertEqual(row['label'][0, 0, 0], 1)
        self.assertTrue(row['legacy'][0, 0, 0, 1])
        self.assertFalse(next_targets(row['label'])[0, 0, 0, 1])

    def test_tail_and_current_outage_masked(self):
        up = np.ones((80, 8, 8), bool)
        up[59:63, 3, 0] = False
        row = timing_labels(up, [60, 64])
        self.assertFalse(row['valid'][0, 0, 0])
        self.assertFalse(row['valid'][1].any())
        self.assertFalse(row['valid'][:, np.arange(5), np.arange(3, 8)].any())

    def test_no_future_input_and_support(self):
        x = np.zeros((120, 5, 8, 48), np.float16)
        m = np.ones_like(x, bool)
        c = np.zeros((120, 5, 8, 9), np.float16)
        x[..., 46] = 1
        before = windows(x, m, c, [60], [[0, 0, 1]])
        x[60:] = 999
        after = windows(x, m, c, [60], [[0, 0, 1]])
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        support = causal_support(x, m, [20, 60])
        self.assertFalse(support[0].any())
        self.assertTrue(support[1, 0, 1])
        self.assertFalse(support[:, np.arange(5), np.arange(3, 8)].any())

    def test_probability_aggregation_and_event_cadence(self):
        p = np.arange(1, 18, dtype=float)
        p /= p.sum()
        result = next_probabilities(p)
        self.assertLessEqual(result[1], min(result[0], result[2]))
        up = np.ones((160, 8, 8), bool)
        up[67:69, 3, 0] = False  # onset at 33.5 s
        times = np.arange(1, 81)
        scores = np.zeros((80, 5, 8))
        scores[29, 0, 0] = 1     # alarm at 30 s
        support = np.ones_like(scores, bool)
        hours = []
        for cadence in [1, 3]:
            row = event_metrics(scores, support, up, times, cadence=cadence)
            self.assertEqual(row['event_hits_3s'], 1)
            self.assertEqual(row['eligible_events'], 1)
            hours.append(row['drone_hours'])
        self.assertEqual(*hours)

    def test_model_gradient_and_checkpoint_round_trip(self):
        import torch
        torch.set_num_threads(1)
        checkpoint=PROJECT/'results/transformer/pretrain/checkpoints/s0002800.pt'
        model=make_model(checkpoint)
        before=model.timing.weight.detach().clone()
        values=torch.zeros((8,40,48))
        masks=torch.ones_like(values,dtype=torch.bool)
        intents=torch.zeros((8,40,9))
        geometry=torch.zeros((8,46));geometry[:,-1]=1
        target=torch.arange(8)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        logits,legacy=model(values,masks,intents,geometry)
        loss=torch.nn.functional.cross_entropy(logits,target)+legacy.mean()*.01
        loss.backward()
        self.assertGreater(float(model.timing.weight.grad.norm()),0.)
        self.assertGreater(float(model.backbone.project[0].weight.grad.norm()),0.)
        optimizer.step()
        self.assertFalse(torch.equal(before,model.timing.weight))
        buffer=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),buffer)
        buffer.seek(0)
        saved=torch.load(buffer,weights_only=True)
        restored=make_model(checkpoint)
        restored.load_state_dict(saved['model'])
        other_optimizer=torch.optim.Adam(restored.parameters(),lr=1e-4)
        other_optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            torch.testing.assert_close(model(values,masks,intents,geometry)[0],restored(values,masks,intents,geometry)[0],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
