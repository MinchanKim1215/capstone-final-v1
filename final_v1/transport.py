"""Causal command loss simulation and receiver-side fallback."""
import numpy as np


class CommandChannel:
    """Command+ACK samples on the last measured simulator RF snapshot.

    A first-relay coordinator floods for at most three hops. This is a
    boundary-time command-channel model with a separate RNG, not TDMA airtime
    scheduling. Out-of-band timing and the shared navigation planner remain
    assumptions of this study.
    """
    def __init__(self, seed, n, ns, baseline_steps):
        self.ns = ns
        self.last = np.zeros(n-ns, dtype=int)
        self.last_tick = np.full(n-ns, -100, dtype=int)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 401, 1]))
        self.windows = [(max(12, int(baseline_steps*fraction)), int((i+1)%(n-ns)))
                        for i, fraction in enumerate((.20, .47, .71))]

    def apply(self, tick, probabilities, requested, observation, local_rf, fallback):
        ns = self.ns
        n = len(probabilities)
        # Use common random numbers across treatments, including where they diverge.
        delivered_edges = self.rng.random((n,n)) < probabilities
        forced = self.rng.random(n-ns) < .03
        for start, relay in self.windows:
            if start <= tick < start+4:
                forced[relay] = True
        forced[0] = False
        delivered_edges[ns+np.flatnonzero(forced)] = False
        delivered_edges[:, ns+np.flatnonzero(forced)] = False
        reachable = np.zeros(n, dtype=bool)
        reachable[ns] = True
        hops = np.full(n, -1, dtype=int)
        hops[ns] = 0
        for hop in range(1, 4):
            new = delivered_edges[reachable].any(axis=0) & ~reachable
            hops[new] = hop
            reachable |= new
        received = reachable[ns:]
        self.last[received] = requested[received]
        self.last_tick[received] = tick
        applied = np.where(received, requested, 0)
        reason = np.where(received, "received", "nominal").astype("<U16")
        if fallback == "local":
            for relay in np.flatnonzero(~received):
                valid = observation["action_mask"][relay]
                old = self.last[relay]
                if tick-self.last_tick[relay] == 1 and valid[old]:
                    applied[relay], reason[relay] = old, "hold_one_cycle"
                    continue
                if not local_rf[relay]:
                    continue
                local = observation["teacher"][relay]
                best = int(np.where(valid, local, -np.inf).argmax())
                if local[best] > local[0]+.4:
                    applied[relay], reason[relay] = best, "local_radio"
        return applied, received, hops[ns:], forced, reason
