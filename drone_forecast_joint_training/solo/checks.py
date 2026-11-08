import importlib.metadata,json,os,platform,sys,time,unittest,shutil
from pathlib import Path
import numpy as np
import torch
from solo.contracts import ROOT,HISTORY,load_plan,sources,sha,atomic_json,seed_all,device_for,deterministic


def toy_batch(n=4,device='cpu'):
    b=dict(context=torch.randn(n,5,91)*.2,candidates=torch.randn(n,5,25,12)*.2,
        mask=torch.ones(n,5,25,dtype=torch.bool),teacher=torch.randn(n,5,25)*.1,gain=torch.randn(n,5,25,8)*.2,
        obs_values=torch.randn(n,5,HISTORY,8,48).half(),obs_masks=torch.ones(n,5,HISTORY,8,48,dtype=torch.bool),
        obs_intents=torch.randn(n,5,HISTORY,8,9).half(),geometry=torch.randn(n,5,8,46).half(),
        prediction_supported=torch.ones(n,5,8,dtype=torch.bool),prediction_targets=(torch.rand(n,5,8,4)>.9).float(),
        prediction_valid=torch.ones(n,5,8,4,dtype=torch.bool),state=torch.randn(n,224)*.2,
        advantage=torch.linspace(-1,1,n),returns=torch.randn(n))
    b['geometry'][...,-1]=1;b['obs_values'][...,46]=1;b['mask'][...,20:]=False
    for r in range(5):b['prediction_supported'][:,r,r+3]=False
    return {k:v.to(device) for k,v in b.items()}


def verify():
    plan=load_plan();groups={}
    for m in plan['maps']:
        if sha(ROOT/m['path'])!=m['sha256']:raise ValueError('Map changed: '+m['key'])
        groups.setdefault(m['joint_spatial_group'],set()).add(m['joint_split'])
        if m['joint_split']=='test' and m['previous_role']!='fresh':raise ValueError('Previously evaluated final-test geography')
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Spatial leakage')
    return dict(passed=True,maps=len(plan['maps']),counts=plan['counts'],spatial_groups=len(groups),source_sha256=sources(),test_opened=(ROOT/'results/transformer/test_access.json').exists())


def doctor():
    if sys.version_info[:2]!=(3,12):raise ValueError('Use Python 3.12 64-bit')
    if sys.maxsize<=2**32:raise ValueError('64-bit Python required')
    if torch.__version__.split('+')[0]!='2.9.1':raise ValueError('Use the pinned PyTorch 2.9.1 build')
    import shapely,numba,scipy
    versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','shapely','numba','scikit-learn']}
    return dict(passed=True,platform=platform.platform(),python=sys.version.split()[0],versions=versions,
        cuda_available=torch.cuda.is_available(),torch_cuda_runtime=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,cpu_count=os.cpu_count(),
        free_disk_gib=shutil.disk_usage(ROOT).free/1024**3,data=verify(),
        note='CPU self-tests do not validate CUDA training; run cuda-check on the target NVIDIA computer')


def training_device_probe(device):
    from solo.model import Controller
    from solo.data import actor_inputs
    from solo.contracts import atomic_torch
    from solo.training import move_optimizer
    import tempfile
    d=device_for(device);deterministic(True);seed_all(261029)
    model=Controller('joint').to(d);model.train();batch=toy_batch(2,d)
    before={n:p.detach().clone() for n,p in model.risk.named_parameters()}
    chosen=torch.zeros((2,5),dtype=torch.long,device=d)
    with torch.no_grad():
        logits,_=model.policy(*actor_inputs(batch));gpu_logp=torch.distributions.Categorical(logits=logits).log_prob(chosen)
        cpu=Controller('joint');cpu.load_state_dict({n:p.detach().cpu() for n,p in model.state_dict().items()})
        cpubatch={n:v.cpu() for n,v in batch.items()};cpulogits,_=cpu.policy(*actor_inputs(cpubatch))
        cpu_logp=torch.distributions.Categorical(logits=cpulogits).log_prob(chosen.cpu())
        parity=float((cpu_logp-gpu_logp.cpu()).abs().max());del cpu
    if parity>2e-4:raise ValueError('CPU/target behavior-policy mismatch: '+str(parity))
    logits,p=model.policy(*actor_inputs(batch))
    policy=-(torch.distributions.Categorical(logits=logits).log_prob(chosen)*batch['advantage'][:,None]).mean()
    parameters=list(model.risk.encoder_parameters())+list(model.risk.pred_head.parameters())
    gradients=torch.autograd.grad(policy,parameters,retain_graph=True,allow_unused=True);cut=len(model.risk.encoder_parameters())
    norm=lambda part:float(torch.sqrt(sum(g.double().square().sum() for g in part if g is not None)).cpu())
    encoder=norm(gradients[:cut]);head=norm(gradients[cut:])
    loss=policy+.1*torch.nn.functional.binary_cross_entropy(p.clamp(1e-7,1-1e-7),batch['prediction_targets'])
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    def update(network,opt,objective):
        opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);opt.step()
    def objective(network):
        logits,p=network.policy(*actor_inputs(batch))
        return -(torch.distributions.Categorical(logits=logits).log_prob(chosen)*batch['advantage'][:,None]).mean()+.1*torch.nn.functional.binary_cross_entropy(p.clamp(1e-7,1-1e-7),batch['prediction_targets'])
    update(model,optimizer,loss)
    if not torch.isfinite(loss) or encoder<=0 or head<=0:raise ValueError('Target Transformer/PPO backward verification failed')
    if not any(not torch.equal(p,before[n]) for n,p in model.risk.named_parameters()):raise ValueError('Target predictor weights did not change')
    with tempfile.TemporaryDirectory(dir=ROOT/'results/transformer') as directory:
        path=Path(directory)/'target_resume.pt';atomic_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        update(model,optimizer,objective(model))
        expected={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
        restored=torch.load(path,map_location='cpu',weights_only=False)
        resumed=Controller('joint').to(d);resumed.load_state_dict(restored['model'])
        opt=torch.optim.Adam([p for p in resumed.parameters() if p.requires_grad],lr=1e-4)
        opt.load_state_dict(restored['optimizer']);move_optimizer(opt,d);update(resumed,opt,objective(resumed))
        equal=all(torch.equal(p.detach().cpu(),expected[n]) for n,p in resumed.state_dict().items())
    if not equal:raise ValueError('Target checkpoint/Adam resume differs from uninterrupted execution')
    if d.type=='cuda':torch.cuda.synchronize(d)
    return dict(passed=True,device=str(d),gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,
        capability=list(torch.cuda.get_device_capability(d)) if d.type=='cuda' else None,
        torch=torch.__version__,cuda_runtime=torch.version.cuda,encoder_policy_gradient=encoder,head_policy_gradient=head,
        predictor_updated=True,cpu_target_logp_max_error=parity,checkpoint_resume_bitwise_equal=equal,
        lightgbm_used=False,gru_used=False)


def cuda_check(device='cuda'):
    if not device.startswith('cuda'):raise ValueError('cuda-check requires a CUDA device')
    return training_device_probe(device)


def selftest():
    started=time.perf_counter();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(ROOT/'results/transformer/verification/selftest.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),wall_s=time.perf_counter()-started,platform=platform.platform(),device='cpu',cuda_tested=False))
    return result.wasSuccessful()
