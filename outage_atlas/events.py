"""Explain complete outage intervals and the transitions that started them."""
import numpy as np

from .radio import TRANSITION, minimal_masks, names, radio_graph
from cooperative.packets import hop_probabilities
from forecast_causal.causes import annotate
from forecast_prevention.diagnostics import connected, outage_runs, reachability


def component_groups(graph):
    return sorted({tuple(np.flatnonzero(row).tolist()) for row in reachability(graph)})


def preceding_terms(loss, extra, noise, shadow, t, ns):
    """Roll individual physical terms back 0.5 s, holding the other terms current."""
    components = (extra, loss-extra, noise, shadow)
    fixes = []
    receive = np.empty_like(loss[t])
    sinr = np.empty_like(receive)
    for mask in range(1, 16):
        values = [a[t-int(bool(mask & (1 << i)))] for i, a in enumerate(components)]
        building, distance, interference, fade = values
        total = loss[t-1] if mask & 3 == 3 else (loss[t] if mask & 3 == 0 else distance+building)
        p = hop_probabilities(total, interference, fade, receive, sinr)
        if connected(radio_graph(p), ns):
            fixes.append(mask)
    if 15 not in fixes:
        raise ValueError('Freezing all terms did not restore the actual preceding frame')
    return [names(mask, TRANSITION) for mask in minimal_masks(fixes)]


def changed_edges(env, positions, graph, p, radio, noise_db, t):
    reach = reachability(graph[t])
    just_lost = graph[t-1] & ~graph[t] & ~reach
    senders, receivers = np.where(np.triu(just_lost, 1))
    edges = []
    for i, j in zip(senders.tolist(), receivers.tolist()):
        trial = graph[t].copy()
        trial[i, j] = trial[j, i] = True
        extra = radio['extra'][t-1:t+1, i, j]
        distance = np.linalg.norm(positions[t-1:t+1, i]-positions[t-1:t+1, j], axis=-1)
        edges.append(dict(nodes=[i,j], role_pair='search-search' if j < env.ns else
                          ('search-relay' if i < env.ns else 'relay-relay'),
                          reconnects_search_alone=bool(connected(trial, env.ns)),
                          distance_m=distance.tolist(), building_loss_db=extra.tolist(),
                          distance_loss_db=(radio['loss'][t-1:t+1,i,j]-extra).tolist(),
                          noise_dbm=noise_db[t-1:t+1][:, [i,j]].tolist(),
                          shadow_loss_db=env.shadow[t-1:t+1,i,j].tolist(),
                          geometric_los=radio['los'][t-1:t+1,i,j].tolist(),
                          minimum_directional_hop_probability=np.minimum(p[t-1:t+1,i,j], p[t-1:t+1,j,i]).tolist()))
    return edges


class Observations:
    def __init__(self, env):
        observer = env.ns
        counts = env.telemetry.counts[:env.t, observer]
        stamps = np.arange(len(counts))*.5
        last = np.maximum.accumulate(np.where(counts>0, stamps[:, None], -np.inf), axis=0)
        self.x = env.telemetry.x[:env.t, observer]
        self.m = env.telemetry.mask[:env.t, observer]
        self.age = stamps[:, None]+.5-last[:, :env.ns]
        self.ns = env.ns

    def at(self, frame):
        if frame < 0:
            return None
        x, m = self.x[frame], self.m[frame]
        rf = m[:,28:31].all(axis=1) & (x[:,30] <= 2.)
        rf[self.ns] = False
        margin = rf & m[:,41]
        return dict(time_s=(frame+1)*.5, local_recent_rf_links=int(rf.sum()),
                    missing_search_beacons=np.flatnonzero(self.age[frame] > 3.).tolist(),
                    never_received_search_beacons=np.flatnonzero(~np.isfinite(self.age[frame])).tolist(),
                    minimum_received_margin_db=float(x[margin,41].min()) if margin.any() else None,
                    local_noise_dbm=float(np.median(x[rf,28]-x[rf,29])) if rf.any() else None,
                    local_queue=int(x[0,11]) if m[0,11] else None)

    def before(self, t):
        when = t*.5
        first = max(7, int((when-7.)//3)+1)
        last = int((when-3.)//3)
        ticks = [self.at(step*6-1) for step in range(first,last+1)]
        return dict(at_3s=self.at(t-7), at_6s=self.at(t-13), decision_window=ticks,
                    qualified_decisions=sum(row['local_recent_rf_links']>0 and not row['missing_search_beacons']
                                            for row in ticks))


def network_events(env, probs, repaired, radio, interference, positions, observations=None):
    graph = radio_graph(probs)
    health = repaired[0]
    phases = [row['phase'] for row in env.history]
    noise_db = 10*np.log10(10**(-107/10)+interference)
    results = []
    for t, stop in outage_runs(health):
        first = [k for k in range(1,8) if repaired[k,t:t+2].all()]
        full = [k for k in range(1,8) if repaired[k,t:stop].all()]
        groups = component_groups(graph[t])
        search_groups = [g for g in groups if g[0] < env.ns]
        row = dict(onset_s=t*.5, duration_s=(stop-t)*.5, left_censored=t==0,
                   right_censored=stop==len(health), phase=phases[t//6],
                   jammer_active=bool(np.any(interference[t] > 0)),
                   components=groups, search_components=len(search_groups),
                   isolated_search_nodes=[g[0] for g in search_groups if len(g)==1],
                   positions_xy_m=positions[t].tolist(),
                   first_second_minimal_repairs=[names(k) for k in minimal_masks(first)],
                   entire_cut_minimal_repairs=[names(k) for k in minimal_masks(full)],
                   restored_fraction_by_removal=[float(repaired[k,t:stop].mean()) for k in range(8)],
                   still_down_without_all_three_seconds=float((~repaired[7,t:stop]).sum()*.5))
        if t:
            row['minimal_prior_term_restorations'] = preceding_terms(radio['loss'], radio['extra'],
                     interference, env.shadow, t, env.ns)
            row['newly_failed_cross_component_edges'] = changed_edges(env, positions, graph, probs, radio, noise_db, t)
        if observations is not None:
            row['observed_before'] = observations.before(t)
        results.append(row)
    ii, jj = np.triu_indices(env.n, 1)
    individual = sum(len(outage_runs(graph[:,i,j])) for i,j in zip(ii,jj))
    return results, individual


def service_events(health, network, packet_evidence):
    output = []
    for start, stop in outage_runs(health):
        when = start*.5
        row = dict(onset_s=when, duration_s=(stop-start)*.5, left_censored=start==0,
                   network_down_at_onset=not bool(network[start]),
                   network_down_any_during_event=bool((~network[start:stop]).any()),
                   network_down_during_settled_birth_window=bool((~network[max(0,start-2):start]).any()))
        if 24 <= when <= len(network)*.5-8.5:
            detail = annotate(packet_evidence, when)
            row.update(flags=detail['flags'], late_packets=detail['bad_packets'],
                       settled_packets=detail['total_settled_packets'],
                       mean_queue_before_deadline_s=detail['mean_queue_before_deadline'],
                       mean_service_before_deadline_s=detail['mean_service_before_deadline'])
        output.append(row)
    return output
