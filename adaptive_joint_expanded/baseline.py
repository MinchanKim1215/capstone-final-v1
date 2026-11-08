"""Private nominal trajectories and paired exogenous interference/shadowing."""
import math
import numpy as np
from . import ARTIFACTS
from .contracts import DEPLOYMENT_BY_KEY, atomic_json, digest, read_json, sha


def ensure_nominal(city, deployment, scenario):
    from cooperative.motion import motion_stream
    spec = dict(map=city.sha256, deployment=deployment.key, direction=scenario.direction,
                engine=sha(__file__), motion='cooperative.motion-v5-unchanged')
    folder = ARTIFACTS/'cache/nominal'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(digest(spec)[:24]+'.npz')
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        item = read_json(meta)
        if item['spec'] != spec or item['sha256'] != sha(path):
            raise ValueError('Private nominal cache failed its content contract')
        return path
    stream = motion_stream(city, deployment.design(), scenario)
    proposal = next(stream)
    nodes = [proposal['nodes'].copy()]
    minimum = math.inf
    terminal = False
    for _ in range(2400):
        event = stream.send(np.zeros((deployment.n, 2)))
        nodes.append(event['new'].copy())
        minimum = min(minimum, event['minimum_separation_m'])
        terminal = bool(event['done'])
        try:
            proposal = next(stream)
        except StopIteration:
            break
    if minimum < 5 - 1e-7:
        raise ValueError('Nominal deployment violates separation: ' + deployment.key)
    temp = path.with_suffix('.pending.npz')
    np.savez_compressed(temp, nodes=np.asarray(nodes, np.int32))
    temp.replace(path)
    atomic_json(meta, dict(spec=spec, sha256=sha(path), completed=terminal,
                           duration_s=(len(nodes)-1)*3., minimum_separation_m=minimum))
    return path


def paired_shadow(seed, nt, n, dt, sigma, tau):
    """Preserve the original eight-agent stream; extend to ten without changing it.

    N=6 takes the same first six radio identities. New identities have independent
    pair-specific streams. Packet draws remain action-dependent legacy draws.
    """
    from cooperative.packets import shadow_sequence
    out = np.zeros((nt, max(8,n), max(8,n)))
    out[:, :8, :8] = shadow_sequence(seed, nt, 8, dt, sigma, tau)
    rho = math.exp(-dt/tau)
    scale = sigma*math.sqrt(1-rho*rho)
    for i in range(8,n):
        for j in range(i):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 827, i, j]))
            z = 0.
            noise = rng.normal(size=nt)
            for t in range(nt):
                z = rho*z + scale*noise[t]
                out[t,i,j] = out[t,j,i] = z
    return np.ascontiguousarray(out[:, :n, :n])
