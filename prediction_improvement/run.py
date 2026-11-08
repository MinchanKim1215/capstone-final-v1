"""CPU-only isolated pilot; does not change the ongoing CUDA/PPO experiment."""
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROJECT = WORKSPACE/'drone_forecast_joint_training'
JOB = HERE/'runs/next_outage_1s_v1'
JOB.mkdir(parents=True, exist_ok=True)
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['NUMBA_CACHE_DIR'] = str(JOB/'numba_cache')
os.environ['PYTHONUTF8'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from core import timing_labels, causal_support, windows, next_targets, next_probabilities, make_model, event_metrics
from solo.metrics import binary_metrics


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace(temp,path)


def replace(temp,path):
    for attempt in range(12):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.05*(attempt+1))


def status(stage, **more):
    row = dict(state='running', stage=stage, updated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
               pid=os.getpid(), cpu_threads=1, cuda_used=False, main_experiment_modified=False, **more)
    write(JOB/'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def protocol():
    from solo.contracts import sources
    bootstrap = load(PROJECT/'results/transformer/bootstrap/complete.json')
    plan = load(PROJECT/'configs/transformer_plan.json')
    entries = {entry['key']:entry for entry in plan['maps']}
    if sources() != bootstrap['source_sha256']:
        raise ValueError('Live source contract differs from the frozen bootstrap')
    pinned = {str(PROJECT/name):digest for name,digest in bootstrap['source_sha256'].items()}
    pinned[str(PROJECT/'configs/transformer_plan.json')] = sha(PROJECT/'configs/transformer_plan.json')
    checkpoint = PROJECT/'results/transformer/pretrain/checkpoints/s0002800.pt'
    pinned[str(checkpoint)] = sha(checkpoint)
    for name in ['core.py','run.py','test_core.py']:
        pinned[str(HERE/name)] = sha(HERE/name)
    episodes = []
    for split in ['train','validation']:
        for prefix, expected in zip(bootstrap[split+'_paths'],bootstrap[split+'_hashes']):
            source = PROJECT/prefix
            meta = load(source.with_suffix('.json'))
            entry = entries[meta['region']]
            if entry['joint_split'] != split or meta['array_sha256'] != expected:
                raise ValueError('Data split or archive digest contract changed')
            episodes.append(dict(prefix=prefix, archive_sha256=expected, split=split,
                                 region=meta['region'], map_path=entry['path'], map_sha256=entry['sha256'],
                                 spatial_group=entry['joint_spatial_group'], direction=meta['direction']))
    train_groups = {ep['spatial_group'] for ep in episodes if ep['split']=='train'}
    validation_groups = {ep['spatial_group'] for ep in episodes if ep['split']=='validation'}
    assert not train_groups & validation_groups
    data = dict(format='isolated-next-outage-pilot-v1', input_pins=pinned, episodes=episodes,
                checkpoint=str(checkpoint), steps=6000, batch_links=256, seed=2026102917,
                learning_rate=.0001, auxiliary_legacy_bce_weight=.25,
                prediction_cadence_s=1, unchanged_action_cadence_s=3,
                timing_classes='16 half-second bins on [0,8) plus no qualifying onset within 8s',
                displayed_bins_s=[[0,2],[2,4],[4,6],[6,8]], minimum_outage_frames=2,
                interpretation='First next persistent directed-link outage under recorded policy; not counterfactual no-reposition risk.',
                tail='Only label fully observed 8-second horizon plus persistence confirmation; do not turn censored tails into negatives.',
                geometry='Recomputed from stored float16 public RF/intent at the current time only, for all cadences consistently.',
                sample='Per-episode class-stratified samples with inverse sampling-probability weights; no future labels passed to the model.',
                source_corpus='192 supervised bootstrap train episodes, 8 previous validation episodes; no final test access.',
                limitations=['Diagnostic validation maps were previously used for checkpoint selection.',
                             'Next-event timing targets differ from the legacy any-event heads; report them separately.',
                             'A 1-second replay is not a validated integration with the live controller.',
                             'CPU-only work may slow simulation collection slightly; no GPU is reserved.',
                             'This pilot does not yet add new PPO data, simulation conditions, event-identity sampling, or change the controller.',
                             'Stored quantized observations cannot exactly reconstruct original float32 geometry; old published AP must not be directly compared.'])
    path = JOB/'protocol.json'
    if path.exists():
        if load(path) != data:
            raise ValueError('Pilot protocol changed; use a new experiment directory')
    else:
        write(path, data)
    for path, expected in pinned.items():
        if sha(path) != expected:
            raise ValueError('Pinned source or checkpoint changed: '+path)
    return data


def prepare(p):
    from swarm_sim.city import City
    from swarm_sim.config import Scenario
    from advance.features import geometry_profile
    cache = JOB/'data'
    cache.mkdir(exist_ok=True)
    signature = sha(JOB/'protocol.json')
    for i, spec in enumerate(p['episodes']):
        destination = cache/(str(i)+'.npz')
        record = destination.with_suffix('.json')
        if destination.exists() and record.exists():
            saved = load(record)
            if saved['protocol_sha256'] != signature or sha(destination) != saved['sha256']:
                raise ValueError('Derived input cache changed')
            continue
        status('prepare_episode', complete=i, total=len(p['episodes']), split=spec['split'], region=spec['region'])
        source = (PROJECT/spec['prefix']).with_suffix('.npz')
        if sha(source) != spec['archive_sha256'] or sha(PROJECT/spec['map_path']) != spec['map_sha256']:
            raise ValueError('Frozen observation archive or map changed')
        with np.load(source, allow_pickle=False) as z:
            x,m,c,links = [z[key] for key in ['rf_values','rf_masks','rf_intents','links']]
        if len(x) != len(links):
            raise ValueError('RF/link timeline mismatch')
        ends = np.arange(2, len(x)+1, 2)
        target = timing_labels(links, ends)
        support = causal_support(x,m,ends)
        city = City(PROJECT/spec['map_path'], Scenario(direction=spec['direction']))
        g = np.zeros((len(ends),5,8,46), np.float16)
        for first in range(0,len(ends),64):
            endpoint = ends[first:first+64]-1
            result = geometry_profile(city,x[endpoint].reshape(-1,48),m[endpoint].reshape(-1,48),c[endpoint].reshape(-1,9))
            g[first:first+len(endpoint)] = result.reshape(-1,5,8,46).astype(np.float16)
        assert np.isfinite(g).all()
        temp = destination.with_name(destination.name+'.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,ends=ends,geometry=g,support=support,links=links,**target)
        replace(temp,destination)
        write(record,dict(protocol_sha256=signature,sha256=sha(destination),spec=spec))
    status('data_prepared', complete=len(p['episodes']), total=len(p['episodes']))


class Pool:
    def __init__(self,p):
        self.p=p
        self.cache=OrderedDict()

    def episode(self,index):
        if index not in self.cache:
            with np.load(JOB/'data'/f'{index}.npz',allow_pickle=False) as z:
                data={key:z[key] for key in z.files}
            source=(PROJECT/self.p['episodes'][index]['prefix']).with_suffix('.npz')
            with np.load(source,allow_pickle=False) as z:
                for key in ['rf_values','rf_masks','rf_intents']:
                    data[key]=z[key]
            keep=data['support']&data['valid']
            rows=np.argwhere(keep)
            classes=data['label'][keep]
            data['rows']=rows
            data['class_rows']=[rows[classes==k] for k in range(17)]
            self.cache[index]=data
            while len(self.cache)>2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def batch(self,data,rows):
        arrays=windows(data['rf_values'],data['rf_masks'],data['rf_intents'],data['ends'],rows)
        t,r,q=rows.T
        return tuple(torch.as_tensor(v) for v in [*arrays,data['geometry'][t,r,q],data['label'][t,r,q],data['legacy'][t,r,q]])

    def sample(self,rng,indices,n):
        data=self.episode(int(rng.choice(indices)))
        sizes=np.asarray([len(rows) for rows in data['class_rows']])
        active=np.flatnonzero(sizes[:16])
        probabilities=np.zeros(17)
        if len(active) and sizes[16]:
            probabilities[active]=.5/len(active)
            probabilities[16]=.5
        else:
            probabilities[sizes>0]=1/np.count_nonzero(sizes)
        if not sizes.sum():
            raise ValueError('No valid samples in a training flight')
        chosen=rng.choice(17,size=n,p=probabilities)
        rows=np.stack([data['class_rows'][k][rng.integers(sizes[k])] for k in chosen])
        importance=(sizes[chosen]/sizes.sum())/probabilities[chosen]
        return self.batch(data,rows),torch.as_tensor(importance,dtype=torch.float32)


def evaluate(model,pool,indices,full=False):
    ys=[];ps=[];nll=[];events=[]
    model.eval()
    with torch.no_grad():
        for index in indices:
            data=pool.episode(index)
            mask=data['support']&data['valid']
            if not full:
                choose=np.zeros(len(mask),bool)
                choose[np.linspace(20,len(mask)-10,min(96,max(1,len(mask)-30))).astype(int)]=True
                mask=mask&choose[:,None,None]
            # A live prediction must not be gated by future label availability.
            # Labels/validity are used only for offline scoring, never as inputs.
            rows=np.argwhere(data['support'] if full else mask)
            all_probability=np.full((*data['support'].shape,17),np.nan,np.float32) if full else None
            for first in range(0,len(rows),256):
                take=rows[first:first+256]
                v,m,c,g,y,_=pool.batch(data,take)
                logits,_=model(v,m,c,g)
                probability=torch.softmax(logits,-1).numpy()
                t,r,q=take.T
                eligible=data['valid'][t,r,q]
                ys.append(next_targets(y.numpy())[eligible])
                ps.append(next_probabilities(probability)[eligible])
                nll.append(torch.nn.functional.cross_entropy(logits,y,reduction='none').numpy()[eligible])
                if full:
                    t,r,q=take.T
                    all_probability[t,r,q]=probability
            if full:
                score=next_probabilities(all_probability)[...,2]
                for cadence in [1,3]:
                    events.append(event_metrics(score,data['support'],data['links'],data['ends']*.5,cadence=cadence))
    y,p=np.concatenate(ys),np.concatenate(ps)
    result=dict(heads={name:binary_metrics(y[:,h],p[:,h]) for h,name in enumerate(['next_onset_lt5s','next_onset_3to4s','next_onset_3to7s','next_onset_0to7s'])},
                timing_nll=float(np.concatenate(nll).mean()))
    if full:
        event_summary={}
        for cadence in [1,3]:
            parts=[row for row in events if row['cadence_s']==cadence]
            row={key:sum(item[key] for item in parts) for key in ['first_alerts','correct_first_alerts','eligible_events','event_hits_3s','drone_hours']}
            row['precision']=row['correct_first_alerts']/row['first_alerts'] if row['first_alerts'] else None
            row['event_recall_3s']=row['event_hits_3s']/max(1,row['eligible_events'])
            row['false_alerts_per_drone_hour']=(row['first_alerts']-row['correct_first_alerts'])/max(1e-12,row['drone_hours'])
            event_summary[str(cadence)]=row
        result['descriptive_alarm_at_half']=event_summary
    return result


def train(p):
    torch.manual_seed(p['seed'])
    rng=np.random.default_rng(p['seed'])
    pool=Pool(p)
    train_indices=[i for i,ep in enumerate(p['episodes']) if ep['split']=='train']
    val_indices=[i for i,ep in enumerate(p['episodes']) if ep['split']=='validation']
    model=make_model(p['checkpoint'])
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    history=[];best=None;done=0
    latest=JOB/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['protocol_sha256']!=sha(JOB/'protocol.json'):
            raise ValueError('Resume contract changed')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        done=ck['step'];history=ck['history'];best=ck['best']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
    else:
        counts=np.full(17,.5)
        for index in train_indices:
            with np.load(JOB/'data'/f'{index}.npz',allow_pickle=False) as data:
                keep=data['valid']&data['support']
                counts+=np.bincount(data['label'][keep],minlength=17)
        with torch.no_grad():
            model.timing.bias.copy_(torch.as_tensor(np.log(counts/counts.sum()),dtype=torch.float32))
        write(JOB/'training_class_prior.json',dict(counts_with_half_smoothing=counts.tolist(),prior=(counts/counts.sum()).tolist()))
    def save(path,step):
        temp=path.with_name(path.name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,best=best,
                        protocol_sha256=sha(JOB/'protocol.json'),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),temp)
        replace(temp,path)
    for step in range(done+1,p['steps']+1):
        model.train()
        (v,m,c,g,y,legacy),weight=pool.sample(rng,train_indices,p['batch_links'])
        logits,old=model(v,m,c,g)
        timing=torch.nn.functional.cross_entropy(logits,y,reduction='none')
        auxiliary=torch.nn.functional.binary_cross_entropy(old.clamp(1e-7,1-1e-7),legacy,reduction='none').mean(-1)
        loss=((timing+p['auxiliary_legacy_bce_weight']*auxiliary)*weight).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite prediction-only loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite predictor gradient')
        optimizer.step()
        if step%100==0:
            status('prediction_training', step=step, target=p['steps'], loss=float(loss.detach()))
        if step%500==0 or step==p['steps']:
            evaluated=evaluate(model,pool,val_indices)
            score=.5*evaluated['heads']['next_onset_lt5s']['average_precision']+.5*evaluated['heads']['next_onset_3to7s']['average_precision']
            improved=best is None or score>best
            history.append(dict(step=step,score=score,metrics=evaluated))
            if improved:
                best=score;save(JOB/'best.pt',step)
            write(JOB/'history.json',history)
            status('prediction_validation',step=step,target=p['steps'],score=score,improved=improved)
        if step%200==0 or step==p['steps']:
            save(latest,step)
    best_state=torch.load(JOB/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best_state['model'])
    status('full_prediction_replay',best_step=best_state['step'])
    result=evaluate(model,pool,val_indices,full=True)
    result.update(best_step=best_state['step'],validation_bias='Development diagnostic maps, not an independent final test.',
                  alarm_warning='Only descriptive threshold .5; not tuned or qualified for operational use.',
                  legacy_comparison_warning='Next-event AP has a different target and input reconstruction; do not compare it directly to published legacy AP.')
    write(JOB/'results.json',result)
    for path,expected in p['input_pins'].items():
        if sha(path)!=expected:
            raise ValueError('Pinned input changed during pilot')
    write(JOB/'status.json',dict(state='completed',stage='completed',updated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                                 cpu_threads=1,cuda_used=False,results=str(JOB/'results.json')))
    print('PREDICTION PILOT COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):
            raise OSError('Below-normal priority could not be set')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('This isolated pilot must not access CUDA')
    import msvcrt
    with (JOB/'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0');lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            print('An existing process owns this pilot; no second run started.',flush=True)
            return
        p=protocol()
        if args.prepare_only:
            write(JOB/'status.json',dict(state='prepared',stage='preflight_complete',cpu_threads=1,cuda_used=False,
                                        train_episodes=192,validation_episodes=8,target_steps=p['steps']))
            print('PREDICTION PILOT PREFLIGHT PASSED',flush=True)
            return
        if (JOB/'results.json').exists():
            print('Pilot already completed; saved results retained.',flush=True)
            return
        status('preflight_complete',train_episodes=192,validation_episodes=8,target_steps=p['steps'])
        prepare(p)
        train(p)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write(JOB/'status.json',dict(state='failed',error=repr(exc),updated_at=datetime.now().astimezone().isoformat(timespec='seconds')))
        traceback.print_exc()
        raise
