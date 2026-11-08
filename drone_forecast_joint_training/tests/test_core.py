import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from solo.contracts import ROOT,HISTORY,atomic_json,atomic_torch,sha,seed_all,rng_state,restore_rng,deterministic,load_plan,sources,ensure_map
from solo.model import Controller,RiskTransformer
from solo.data import actor_inputs,prediction_loss
from solo.training import DEFAULTS,make_optimizer,ppo_update,commit,latest
from solo.checks import toy_batch


class ToyData:
    def __init__(self,model,n=4):
        self.data=toy_batch(n);self.value_scale=1.
        with torch.no_grad():
            logits,_=model.policy(*actor_inputs(self.data));distribution=torch.distributions.Categorical(logits=logits)
            chosen=distribution.sample();self.data.update(old_logits=logits.clone(),chosen=chosen,logp=distribution.log_prob(chosen))
    def __len__(self):return len(self.data['returns'])
    def batch(self,indices,device='cpu'):return {k:v[indices].to(device) for k,v in self.data.items()}


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):deterministic(True)
    def setUp(self):seed_all(99103)

    def test_no_recurrent_modules_or_tree_inputs(self):
        import inspect
        model=Controller('joint')
        self.assertFalse(any(isinstance(m,(torch.nn.GRU,torch.nn.LSTM,torch.nn.RNN)) for m in model.modules()))
        self.assertNotIn('base',inspect.signature(model.risk.forward).parameters)
        b=toy_batch(2);self.assertNotIn('base_prediction',b)

    def test_train_eval_dropout_free_parity(self):
        model=Controller();b=toy_batch(2)
        with torch.no_grad():
            model.eval();a,p=model.policy(*actor_inputs(b));model.train();c,q=model.policy(*actor_inputs(b))
        torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-6);torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-6)

    def test_all_padding_is_finite_and_self_masked(self):
        model=Controller();b=toy_batch(2);b['obs_masks'].zero_();b['prediction_supported'].zero_()
        logits,p=model.policy(*actor_inputs(b));self.assertTrue(torch.isfinite(logits).all());self.assertTrue(torch.isfinite(p).all())

    def test_unknown_sensor_values_are_ignored(self):
        model=Controller();b=toy_batch(2);b['obs_masks'][...,:5]=False
        with torch.no_grad():
            a,p=model.policy(*actor_inputs(b));b['obs_values'][...,:5]=1234;c,q=model.policy(*actor_inputs(b))
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(p,q))

    def test_peer_and_time_axes(self):
        model=RiskTransformer();b=toy_batch(2);perm=torch.tensor([2,5,1,0,7,4,6,3])
        with torch.no_grad():
            z,p=model(b['obs_values'],b['obs_masks'],b['obs_intents'],b['geometry'])
            zz,pp=model(b['obs_values'][...,perm,:],b['obs_masks'][...,perm,:],b['obs_intents'][...,perm,:],b['geometry'][...,perm,:])
        torch.testing.assert_close(pp,p[...,perm,:],atol=2e-6,rtol=2e-6)

    def test_probability_order_and_action_mask(self):
        model=Controller();b=toy_batch(2);logits,p=model.policy(*actor_inputs(b))
        self.assertTrue((p[...,1]<=p[...,0]).all());self.assertTrue((p[...,1]<=p[...,2]).all())
        self.assertTrue((p[...,0]<=p[...,3]).all());self.assertTrue((p[...,2]<=p[...,3]).all())
        self.assertTrue(((p>0)&(p<1)).all());self.assertTrue((logits[...,20:]<-1e8).all())

    def test_privileged_state_and_future_labels_do_not_enter_actor(self):
        model=Controller();b=toy_batch(2)
        with torch.no_grad():
            a,p=model.policy(*actor_inputs(b));b['state']+=100;b['prediction_targets']=1-b['prediction_targets'];b['returns']*=100;c,q=model.policy(*actor_inputs(b))
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(p,q))

    def test_history_never_reads_future(self):
        from solo.world import history
        a=np.random.randn(180,5,8,48).astype(np.float16);b=a.copy();b[60:]=100
        np.testing.assert_array_equal(history(a,10),history(b,10));self.assertEqual(history(a,10).shape,(5,40,8,48))
        self.assertTrue((history(a,0)==0).all())

    def test_labels_match_frozen_historical_definition(self):
        from advance.features import targets
        from solo.targets import prediction_targets
        up=np.random.default_rng(8).random((180,8,8))>.2;y,valid=prediction_targets(up,30);rows=[];ys=[]
        for d in range(1,26):
            for r in range(5):
                for p in range(8):
                    if valid[d,r,p].all():rows.append((6*d-1,r+3,p));ys.append(y[d,r,p])
        old=targets(up,np.zeros_like(up,dtype=float),np.asarray(rows))[:,[0,3,4,6]]
        np.testing.assert_array_equal(np.asarray(ys),old)

    def test_tail_censor_and_current_down_links(self):
        from solo.targets import prediction_targets
        up=np.ones((60,8,8),bool);up[:,3,0]=False;y,valid=prediction_targets(up,10)
        self.assertFalse(valid[0].any());self.assertFalse(valid[-1].any());self.assertFalse(valid[:,0,0].any())
        for r in range(5):self.assertFalse(valid[:,r,r+3].any())

    def test_joint_ppo_gradients_update_encoder_and_probability_head(self):
        m=Controller('joint');d=ToyData(m);config=dict(DEFAULTS,mode='joint',epochs=1,minibatch=2)
        stats=ppo_update(m,make_optimizer(m,config),d,config,1,'cpu')
        for k in ['encoder_from_policy_max','predictor_from_policy_max']:self.assertGreater(stats['gradient_proof'][k],0)
        self.assertGreater(stats['encoder_change_l2'],0);self.assertGreater(stats['predictor_change_l2'],0)

    def test_fixed_predictor_does_not_change(self):
        m=Controller('fixed');d=ToyData(m);config=dict(DEFAULTS,mode='fixed',epochs=1,minibatch=2)
        stats=ppo_update(m,make_optimizer(m,config),d,config,1,'cpu')
        self.assertEqual(stats['encoder_change_l2'],0);self.assertEqual(stats['predictor_change_l2'],0)

    def test_exact_cpu_model_optimizer_rng_resume(self):
        m=Controller();config=dict(DEFAULTS,mode='joint',epochs=1,minibatch=2);o=make_optimizer(m,config)
        ppo_update(m,o,ToyData(m),config,1,'cpu')
        saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=rng_state()))
        ppo_update(m,o,ToyData(m),config,2,'cpu');expected=copy.deepcopy((m.state_dict(),o.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';atomic_torch(path,saved);saved=torch.load(path,weights_only=False)
        b=Controller();b.load_state_dict(saved['model']);opt=make_optimizer(b,config);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        ppo_update(b,opt,ToyData(b),config,2,'cpu');self.assertTrue(same(expected,(b.state_dict(),opt.state_dict())))
        atomic_json(ROOT/'results/transformer/verification/exact_cpu_resume.json',dict(passed=True,model_bitwise_equal=True,optimizer_bitwise_equal=True,updates=2))

    def test_stale_behavior_is_rejected(self):
        m=Controller();d=ToyData(m);d.data['logp']+=.1;c=dict(DEFAULTS,mode='joint')
        with self.assertRaisesRegex(ValueError,'Behavior-policy mismatch'):ppo_update(m,make_optimizer(m,c),d,c,1,'cpu')

    def test_atomic_checkpoint_hash_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/transformer') as d:
            path=commit(d,dict(test=1),0);self.assertEqual(latest(d)[1]['test'],1)
            with path.open('ab') as f:f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):latest(d)

    def test_spatial_groups_and_six_training_conditions(self):
        from solo.checks import verify
        from solo.features import training_specs
        r=verify();self.assertTrue(r['passed']);self.assertEqual(r['maps'],245)
        p=load_plan();rows=training_specs(p,7,0,p['counts']['train']*6);g={}
        for r in rows:g.setdefault(r['region'],set()).add((r['kind'],r['direction']))
        self.assertTrue(all(len(v)==6 for v in g.values()))

    def test_holdout_requires_full_validation(self):
        from solo.evaluate import require_test
        with self.assertRaisesRegex(ValueError,'requires'):require_test(ROOT/'not_used.pt',None)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pilot.json';atomic_json(p,dict(format='standalone-transformer-selection-v5',validation_cases=8))
            with self.assertRaisesRegex(ValueError,'full validation'):require_test(ROOT/'not_used.pt',p)

    def test_vectorized_alarm_metrics_equal_reference(self):
        from solo.metrics import alert_metrics,alarm_trials
        rng=np.random.default_rng(33);e=[dict(prediction=rng.random((40,5,8,4)),prediction_supported=rng.random((40,5,8))>.1,links=rng.random((240,8,8))>.06)]
        thresholds=[.1,.3,.7];exits=[.025,.15,.7]
        for row,t,x in zip(alarm_trials(e,2,thresholds,exits,8),thresholds,exits):
            for k,v in alert_metrics(e,2,t,x,8).items():self.assertAlmostEqual(row[k],v)

    def test_no_labels_loss_is_finite(self):
        m=Controller();b=toy_batch(2);b['prediction_valid'].zero_();_,p=m.policy(*actor_inputs(b));loss,n=prediction_loss(p,b)
        self.assertEqual(n,0);self.assertEqual(float(loss.detach()),0)

    def test_stream_requires_correct_time_intervals(self):
        from solo.deploy import StreamingController
        c=StreamingController.__new__(StreamingController);c.last_time=0.
        with self.assertRaisesRegex(ValueError,'0.5-second'):c.observe(3,None,None,None)

    def test_reward_prioritizes_communication(self):
        from cooperative.continuity_world import reward_terms
        m,_=reward_terms(np.ones(6),0,1,100,0,False);o,_=reward_terms(np.zeros(6),0,1,0,0,False)
        self.assertEqual(m[3],-.003);self.assertEqual(o[0],-3.)

    def test_world_never_constructs_lightgbm_or_gru(self):
        from solo.world import World,live_history,supported
        from solo.features import observe_physics,geometry
        entry=next(m for m in load_plan()['maps'] if m['key']=='ansan_bono');ensure_map(entry)
        with patch('cooperative.world.ControlPredictor',side_effect=AssertionError('LightGBM forbidden')),patch('torch.nn.GRU',side_effect=AssertionError('GRU forbidden')):
            env=World(region='ansan_bono',seed=123,kind='clear',direction='east_west',max_steps=8);m=Controller()
            from cooperative.continuity_control import observe as original_observe
            env.risk[:]=.99;env.early[:]=.99
            a=observe_physics(env);b=original_observe(env,False,'probability')
            for aa,bb in zip(a[:4],b):np.testing.assert_allclose(aa,bb,atol=1e-6)
            while not env.done:
                c,f,mask,t,gain=observe_physics(env);g=geometry(env);support=supported(env.last_x[3:8],env.last_mask[3:8],len(env.recorder.values))
                args=[torch.from_numpy(a) for a in [c,f,mask,t,gain,*live_history(env.recorder),g,support]]
                with torch.no_grad():m.policy(*args)
                env.step(np.zeros((8,2)))
            self.assertEqual(len(env.recorder.values),env.k*6);self.assertEqual(env.forecast_calls,0)
