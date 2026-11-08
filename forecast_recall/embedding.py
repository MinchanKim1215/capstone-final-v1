"""Extract actual frozen service input; hooks removed after every call."""
import numpy as np
import torch
from . import PARENT
from .contracts import read,sha
from forecast_graph.inference import network
from forecast_graph.model import tensors
DIM=481


def members():return [m['path'] for m in read(PARENT/'models/v12_mission/model.json')['members']]


def extract(arrays,device='cuda',batch=128):
    outputs=[];logits=[];torch.set_num_threads(2)
    for path in members():
        net=network(path,device);captured=[];parts=[]
        def capture(module,args):captured.append(args[0].detach().cpu().float().numpy())
        handle=net.service.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for start in range(0,len(arrays['n']),batch):
                    ix=np.arange(start,min(start+batch,len(arrays['n'])))
                    result=net(**tensors(arrays,ix,'hybrid_shared',device));parts.append(result['service'].cpu().float().numpy())
        finally:handle.remove()
        value=np.concatenate(captured)
        if value.shape!=(len(arrays['n']),DIM):raise ValueError('Frozen service representation changed')
        outputs.append(value);logits.append(np.concatenate(parts))
    return dict(embedding=np.stack(outputs,1).astype(np.float32),logits=np.stack(logits,1).astype(np.float32))
