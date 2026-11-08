"""Meaningful causal-boundary, missing-message and graph-equivariance checks."""
import numpy as np
import torch
from . import ARTIFACTS
from .contracts import source_records,read,write,now
from .features import LocalHistory,ReceivedTrace,build
from .model import ForecastGraph,tensors
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context


def main():
    r=source_records()[0];ep=Episode(r)
    with np.load(ARTIFACTS/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'][:40].copy();rx=a['rx'][:40].copy();n=int(a['n']);ns=int(a['ns'])
    arrays=[np.array(v[:80]) for v in (ep.values,ep.masks,ep.intents)]
    city,_=public_context(r);city.radio.enable_cache()
    local=LocalHistory(*arrays,n,ns);trace=ReceivedTrace(tx,rx,n,ns);q=np.array([[40,0],[42,0]])
    f=build(local,trace,q,city)
    for value in f.values():assert np.isfinite(value).all()
    altered=[a.copy() for a in arrays];altered[0][42:]=30000;altered[1][42:]=False;altered[2][42:]=-10000
    changed_tx=tx.copy();changed_tx[21:]=0;changed_rx=rx.copy();changed_rx[21:]=0
    future=build(LocalHistory(*altered,n,ns),ReceivedTrace(changed_tx,changed_rx,n,ns),q,city)
    for key in f:np.testing.assert_equal(f[key],future[key])
    #Mutating a current message that was not received must not affect this relay.
    mask=np.unpackbits(rx,axis=-1)[...,:n];unreceived=[j for j in range(n) if j!=ns and not mask[20,ns,j]]
    checked=False
    if unreceived:
        c=tx.copy();c[20,unreceived[0]]=0
        hidden=build(local,ReceivedTrace(c,rx,n,ns),[[42,0]],city)
        for key in f:np.testing.assert_equal(f[key][1:],hidden[key])
        checked=True
    else:
        #Create one missing-message boundary, then vary only its hidden payload.
        j=0;c=rx.copy();c[20,ns,j//8]&=np.uint8(255^(128>>(j%8)))
        a=build(local,ReceivedTrace(tx,c,n,ns),[[42,0]],city);b=tx.copy();b[20,j]=0
        d=build(local,ReceivedTrace(b,c,n,ns),[[42,0]],city)
        for key in a:np.testing.assert_equal(a[key],d[key])
        checked=True
    sample=dict(f,n=np.full(2,n),ns=np.full(2,ns),observer=np.full(2,ns))
    torch.set_num_threads(2);torch.manual_seed(1104);net=ForecastGraph('hybrid_shared').eval()
    inputs=tensors(sample,np.arange(2),'hybrid_shared','cpu')
    with torch.no_grad():out=net(**inputs)
    permutation=np.r_[np.arange(ns)[::-1],np.arange(ns,n)[::-1],np.arange(n,16)]
    p={k:v.copy() for k,v in sample.items()}
    for key in ('node_local','node_shared'):p[key]=p[key][:,:,permutation]
    for key in ('edge_local','edge_shared'):p[key]=p[key][:,:,permutation][:,:,:,permutation]
    p['physics']=p['physics'][:,:,permutation][:,:,:,permutation]
    p['observer']=np.full(2,int(np.flatnonzero(permutation==ns)[0]))
    with torch.no_grad():po=net(**tensors(p,np.arange(2),'hybrid_shared','cpu'))
    np.testing.assert_allclose(out['service'].numpy(),po['service'].numpy(),atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(out['link'].numpy()[:,permutation],po['link'].numpy(),atol=2e-6,rtol=2e-6)
    net.train();result=net(**inputs);loss=sum(v.square().mean() for v in result.values());loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    value=dict(passed=True,created_at=now(),source=r['key'],future_perturbation_exact=True,
        unreceived_payload_does_not_leak=checked,role_preserving_graph_permutation_equivariant=True,
        finite_forward_backward=True,payload_bytes=128,feature_shapes={k:list(v.shape) for k,v in f.items()})
    write(ARTIFACTS/'preflight.json',value);print(value)


if __name__=='__main__':main()
