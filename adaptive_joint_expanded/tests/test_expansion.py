import unittest
from collections import defaultdict
from adaptive_joint_expanded.contracts import plan
from adaptive_joint_expanded.experiment import validation_specs
from adaptive_joint_expanded.initialization import parent_finished


class ExpansionTests(unittest.TestCase):
    def test_all_new_maps_are_registered_and_old_test_is_preserved(self):
        p=plan();new=[m for m in p['maps'] if m.get('source_dataset','legacy_v6')!='legacy_v6']
        self.assertEqual(len(new),100)
        self.assertEqual(sum(m['joint_split']=='test' for m in p['maps']),32)
        self.assertTrue(all(m['source_scene_bytes_preserved'] for m in new))

    def test_spatial_components_do_not_cross_training_and_heldout(self):
        groups=defaultdict(set)
        for m in plan()['maps']:
            if m['joint_split']=='quarantine':continue
            groups[m['joint_spatial_group']].add('train' if m['joint_split']=='train' else 'heldout')
        self.assertTrue(all(len(v)==1 for v in groups.values()))

    def test_small_validation_includes_added_and_original_maps(self):
        p=plan();maps={m['key']:m for m in p['maps']}
        for scope in ('selection','calibration','confirmation'):
            s=validation_specs(8,scope)
            sources=[maps[r['region']].get('source_dataset','legacy_v6')!='legacy_v6' for r in s]
            self.assertEqual(len(s),8)
            self.assertTrue(any(sources));self.assertFalse(all(sources))
            self.assertEqual(len({r['region'] for r in s}),8)

    def test_running_or_failed_parent_cannot_trigger_next_training(self):
        for state in ('running','failed','interrupted','starting'):
            self.assertFalse(parent_finished(dict(state=state,stage='finished')))
        for state in ('completed','early_stopped'):
            self.assertTrue(parent_finished(dict(state=state,stage='finished')))

    def test_actual_world_resolves_added_map_in_its_own_study(self):
        from adaptive_joint_expanded.world import World
        from adaptive_joint_expanded.observation import Observer
        from adaptive_joint_expanded import ARTIFACTS
        region=next(m for m in plan()['maps'] if m.get('source_dataset','legacy_v6')!='legacy_v6' and m['joint_split']=='train')
        world=World(region=region['key'],observer=Observer(),max_steps=1)
        self.assertEqual(world.region,region['key'])
        self.assertEqual(str(world.map_path.resolve()),region['path'])
        import adaptive_joint_expanded.world as module
        self.assertEqual(module.ARTIFACTS,ARTIFACTS)
        self.assertEqual(module.entry_for.__module__,'adaptive_joint_expanded.contracts')


if __name__=='__main__':unittest.main()
