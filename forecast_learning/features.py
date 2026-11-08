"""Past-only RF and prospective cut features shared by training and control."""
import numpy as np

from adaptive_joint_expanded.features import THRESHOLD, widest
from forecast_avoidance.controller import TIMES, samples, onset_mask
from forecast_shield.forecaster import EARLY, evaluate_terms, geometry, hop_margin
from .paths import continuations


PREFIX = ('local_rf_count','missing_search','known_receiver_count','noise_max','noise_mean','noise_trend_max',
          'local_queue','relay_moving_fraction','search_moving_fraction','n_search','n_relay','phase')
TRACE = ('net','no_building','fixed_building','second_pair','weak_links')
SUMMARY = ('shield','short','building','building_change','v20_onset','early_min','early_persistent_min',
           'early_cut_frames','now_margin','first_3s_min')


def names(kind):
    return tuple(f'{kind}_{field}_{i}' for field in TRACE for i in range(len(TIMES)))+tuple(f'{kind}_{k}' for k in SUMMARY)


HELD_COLUMNS = PREFIX+names('held')
COLUMNS = HELD_COLUMNS+names('release')+('release_diff_6s','release_diff_9s')


def describe(xy, loss, extra, state, ns, warning):
    residual = np.nan_to_num(np.clip(state.power-(10.-loss[0]),-5.,5.))
    directed = 10.-loss + residual*np.exp(-TIMES[:,None,None]/4.)-state.noise[None,None,:]-THRESHOLD
    directed -= .25*np.maximum(0.,TIMES[:,None,None]-3.)
    graph = hop_margin(directed)
    pair = widest(graph,ns)
    net = pair.min(axis=1)
    second = np.partition(pair,1,axis=1)[:,1]
    no_build = widest(hop_margin(directed+extra),ns).min(axis=1)
    freeze = widest(hop_margin(directed+extra-extra[0]),ns).min(axis=1)
    i,j = np.triu_indices(graph.shape[-1],1)
    weak = (graph[:,i,j]<3.).mean(axis=1)
    early = net[6:14]
    persistent = np.maximum(net[1:-1],net[2:])[EARLY]
    vector = np.r_[net,no_build,freeze,second,weak,warning.shield_score,warning.short_score,
                   warning.building_support,warning.changing_building_support,warning.onset_in_s,
                   early.min(),persistent.min(),(early<0).sum(),net[0],net[1:7].min()]
    return vector,net,graph


def extract(planner, actions, local_values, phase):
    """Input consists of current motion planning and actual received reports."""
    state, ns = planner.state, planner.design.n_search
    held, release = continuations(planner, actions)
    blocks, margins, graphs, warnings = [], [], [], []
    for knots in (held,release):
        if knots is release and len(blocks) and np.array_equal(held,release):
            blocks.append(blocks[0]); margins.append(margins[0]); graphs.append(graphs[0]); warnings.append(warnings[0])
            continue
        xy = samples(knots)
        loss,extra = geometry(planner.city,planner.scenario,xy)
        warning = evaluate_terms(loss,extra,state,ns,0.)
        block,margin,graph = describe(xy,loss,extra,state,ns,warning)
        blocks.append(block); margins.append(margin); graphs.append(graph); warnings.append(warning)
    moving = (held[1]!=held[0]).any(axis=-1)
    prefix = [np.isfinite(state.power[:,ns]).sum(), (~state.known[:ns]).sum(),
              np.isfinite(state.power).any(axis=0).sum(),state.noise.max(),state.noise.mean(),
              state.noise_trend.max(),local_values[0,11],moving[ns:].mean(),moving[:ns].mean(),
              ns,len(actions),phase]
    dist = np.linalg.norm(held[2:]-release[2:],axis=-1).mean(axis=1)
    x = np.asarray(np.r_[prefix,*blocks,dist],np.float32)
    if x.shape!=(len(COLUMNS),) or np.isinf(x).any():
        raise ValueError('Causal feature contract changed')
    return x,warnings[0],np.asarray(margins),np.asarray(graphs),held[1]
