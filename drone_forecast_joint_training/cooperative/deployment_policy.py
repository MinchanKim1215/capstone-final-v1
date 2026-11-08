"""두 관측/교사 계약을 보존한 분산 정책 확률 혼합. 미래 정답 입력 없음."""
from pathlib import Path
import numpy as np
import torch
from cooperative.policy import load_policy
from swarm_sim.regions import ROOT,sha256_file

FORMAT='paired-teacher-consensus-v3.1'


def mixture_action(old_logits,new_logits,mask,alpha):
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Finite mixture weight in [0,1] required')
    mask=torch.as_tensor(mask,dtype=torch.bool)
    if mask.shape!=(5,25) or not mask.any(-1).all():raise ValueError('Each relay needs a valid action')
    if old_logits.shape!=mask.shape or new_logits.shape!=mask.shape or not torch.isfinite(old_logits[mask]).all() or not torch.isfinite(new_logits[mask]).all():
        raise ValueError('Finite logits matching the action mask required')
    old=torch.softmax(old_logits.masked_fill(~mask,-torch.inf),-1)
    new=torch.softmax(new_logits.masked_fill(~mask,-torch.inf),-1)
    probs=(1-alpha)*old+alpha*new
    chosen=probs.argmax(-1)
    return chosen.cpu().numpy(),probs.gather(-1,chosen[:,None]).squeeze(-1).clamp_min(1e-30).log().cpu().numpy()


class ConsensusPolicy:
    def __init__(self,old,new,alpha):self.old,self.new,self.alpha=old,new,alpha
    def action_pair(self,old_obs,new_obs):
        if not np.array_equal(old_obs[2],new_obs[2]):raise ValueError('Consensus masks differ')
        with torch.no_grad():
            def logits(net,obs):return net.actor(torch.as_tensor(obs[0]),torch.as_tensor(obs[1]),
                torch.as_tensor(obs[2]),torch.as_tensor(obs[3],dtype=torch.float32))
            return mixture_action(logits(self.old,old_obs),logits(self.new,new_obs),old_obs[2],self.alpha)


def load_consensus(path):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False)
    if ck.get('format')!=FORMAT or ck.get('teacher_profile') not in ['linear','probability']:
        raise ValueError('Unsupported paired teacher policy')
    if not np.isfinite(ck['alpha']) or not 0<=ck['alpha']<=1:raise ValueError('Invalid mixture weight')
    nets=[]
    for name in ['old_policy','priority_policy']:
        item=ck[name];file=ROOT/item['path']
        if sha256_file(file)!=item['sha256']:raise ValueError('Consensus member checkpoint changed')
        net,meta=load_policy(file)
        if not meta.get('use_risk',True):raise ValueError('Consensus requires two risk-enabled parents')
        if name=='priority_policy' and meta.get('teacher_profile')!=ck['teacher_profile']:
            raise ValueError('Consensus teacher profile changed')
        if name=='old_policy' and meta.get('continuity_version'):raise ValueError('Old teacher member is not a v2 policy')
        nets.append(net)
    return ConsensusPolicy(*nets,ck['alpha']),ck
