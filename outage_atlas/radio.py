"""Replay actual physical terms using the same hop implementation and packet model."""
from contextlib import contextmanager
from itertools import combinations

import numpy as np
from numba import njit

from adaptive_joint_expanded.contracts import DEPLOYMENT_BY_KEY
from cooperative.packets import PacketSim, hop_probabilities
from forecast_prevention.diagnostics import connected, outage_runs
from forecast_service.replay import service_health


FACTORS = ('building', 'interference', 'shadow')
TRANSITION = ('building_change', 'distance_change', 'interference_change', 'shadow_change')


class GeometryTrace:
    """Read the first two original propagation calls in each motion step."""

    def __init__(self, env):
        self.env = env
        self.loss, self.extra, self.los, self.jammer = [], [], [], []
        self.calls = 0

    @contextmanager
    def record(self):
        radio = self.env.city.radio
        original = radio.propagation
        n = self.env.n

        def propagation(tx, rx, scenario):
            values = original(tx, rx, scenario)
            self.calls += 1
            if self.calls == 1:
                if tx.shape != rx.shape or tx.shape != (6*n*n, 3):
                    raise ValueError('Expected actual air-to-air propagation first')
                loss, los, extra = values
                self.loss.append(loss.reshape(6, n, n))
                self.extra.append(extra.reshape(6, n, n))
                self.los.append(los.reshape(6, n, n))
            elif self.calls == 2:
                if not np.array_equal(tx, self.env.jammer[None]) or rx.shape != (6*n, 3):
                    raise ValueError('Expected actual jammer propagation second')
                self.jammer.append(values[0].reshape(6, n))
            return values

        radio.propagation = propagation
        try:
            yield
        finally:
            radio.propagation = original

    def arrays(self):
        return {name: np.concatenate(getattr(self, name)) for name in ('loss', 'extra', 'los', 'jammer')}


@njit(cache=True)
def probabilities(loss, interference, shadow):
    frames, n, _ = loss.shape
    output = np.empty_like(loss)
    received = np.empty((n, n))
    sinr = np.empty((n, n))
    for t in range(frames):
        output[t] = hop_probabilities(loss[t], interference[t], shadow[t], received, sinr)
    return output


def radio_graph(probs):
    return (probs >= .9) & (probs.swapaxes(-2, -1) >= .9)


def interference_regimes(env, jammer_loss, nominal):
    anchor = nominal(env.city, DEPLOYMENT_BY_KEY['v8_s3_r5_40'], env.scenario)
    with np.load(anchor, allow_pickle=False) as data:
        anchor_duration = (len(data['nodes']) - 1)*3.
    rng = np.random.default_rng(env.seed)
    rng.integers(len(env.city.target_points))  # Match the original jammer location draw.
    start = float(rng.uniform(.15, .45)*anchor_duration)
    stop = float(rng.uniform(.65, .9)*anchor_duration)
    power = float(rng.uniform(-3., 3.)) if env.kind == 'clear' else env.jam_power
    if env.kind == 'intermittent' and (start != env.jam_start or stop != env.jam_end):
        raise ValueError('Paired interference timing changed')
    active = 10 ** ((power - jammer_loss) / 10)
    times = np.arange(len(jammer_loss))*.5
    intermittent = (times >= start) & (times < stop)
    return dict(clear=np.zeros_like(active), intermittent=active*intermittent[:, None], continuous=active), dict(
        jammer_xy_m=env.jammer[:2].tolist(), tx_power_dbm=power, intermittent_start_s=start, intermittent_stop_s=stop)


def minimal_masks(mask):
    return [k for k in sorted(mask) if not any(j != k and j & k == j for j in mask)]


def names(mask, factors=FACTORS):
    return [name for i, name in enumerate(factors) if mask & (1 << i)]


def scenarios(env, radio, interference):
    loss, extra = radio['loss'], radio['extra']
    shadow = env.shadow[:len(loss)]
    zero_shadow = np.zeros_like(shadow)
    zero_noise = np.zeros_like(interference['clear'])
    # These cases recur across regimes, including when interference is removed.
    clear = {}
    for building in (False, True):
        for fading in (False, True):
            p = probabilities(loss-extra if building else loss, zero_noise, zero_shadow if fading else shadow)
            clear[int(building) | (4*int(fading))] = (p, connected(radio_graph(p), env.ns))
    for kind, noise in interference.items():
        repaired = np.empty((8, len(loss)), bool)
        for mask in range(8):
            key = mask & 5
            if kind == 'clear' or mask & 2:
                p, repaired[mask] = clear[key]
            else:
                p = probabilities(loss-extra if mask & 1 else loss, noise, zero_shadow if mask & 4 else shadow)
                repaired[mask] = connected(radio_graph(p), env.ns)
            if mask == 0:
                base = p
        yield kind, base, repaired


def packets_for(env, probs, nodes):
    packet = PacketSim(env.n, env.ns, env.city.visible, .5, env.seed + 1000003,
                       max_duration_s=env.t*.5+3, reserved_slots=env.n)
    for t, p in enumerate(probs):
        packet.interval(p, nodes[t//6])
    packet.drain()
    return packet


def packet_summary(packet, frames):
    generated = int(packet.ic[0])
    healthy = service_health(packet.born[:generated], packet.status[:generated],
                             packet.delays[:generated], frames)['service_healthy']
    if generated != int(packet.ic[1]+packet.ic[5]+packet.ic[6]+packet.ic[7]):
        raise ValueError('Replay packet conservation failed')
    return healthy, dict(generated_packets=generated, delivered_packets=int(packet.ic[1]),
                         timely_pdr=float(packet.ic[8]/max(1, generated)), retries=int(packet.ic[4]),
                         ttl_drops=int(packet.ic[5]), retry_drops=int(packet.ic[6]), overflow_drops=int(packet.ic[7]),
                         service_runs=len(outage_runs(healthy)))
