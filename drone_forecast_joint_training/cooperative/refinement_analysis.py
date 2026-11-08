"""성능 재집계, 공간 그룹 대응 구간, 저장 궤적에 대한 독립 검수."""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
import torch

from cooperative.analyze import key, summarize, METRICS
from cooperative.io import validated_result
from cooperative.refine import STUDY, MODELS, atomic_json, protocol
from swarm_sim.config import Scenario
from swarm_sim.regions import ROOT, sha256_file
from usable.collect import prepare_city_without_unused_radio_cache

NETWORK_COLUMNS={'all_drones_connected_mean':2,'mean_components':3,'largest_component_mean':4,
                 'mean_degree':5,'isolated_drone_fraction':6,'network_efficiency_mean':8,
                 'search_shortest_hops_mean':9}


def graph_statistics_from_links(links):
    """8노드 무방향 링크의 전체 10개 지표를 원래 Numba 엔진과 별도로 계산."""
    links=np.asarray(links,bool)
    if links.ndim!=3 or links.shape[1:]!=(8,8) or not np.array_equal(links,links.transpose(0,2,1)):
        raise ValueError('8노드 대칭 링크 기록 필요')
    if np.any(links[:,np.arange(8),np.arange(8)]):raise ValueError('자기 링크는 비가용이어야 합니다.')
    hops=np.where(links,1.,1e9);hops[:,np.arange(8),np.arange(8)]=0
    for via in range(8):hops=np.minimum(hops,hops[:,:,via,None]+hops[:,None,via,:])
    reach=hops<1e8;sizes=reach.sum(-1);pair=reach[:,[0,0,1],[1,2,2]]
    inverse=np.divide(1.,hops,out=np.zeros_like(hops),where=(hops>0)&reach)
    ph=hops[:,[0,0,1],[1,2,2]]
    return np.column_stack([pair.all(1),pair.mean(1),reach.all((1,2)),
                            np.rint((1./sizes).sum(1)),sizes.max(1)/8.,links.sum((1,2))/8.,
                            (links.sum(-1)==0).mean(1),links.sum((1,2))/56.,inverse.sum((1,2))/56.,
                            np.where(pair,ph,0).sum(1)/np.maximum(pair.sum(1),1)])


def paired_spatial_interval(a, b, metric, map_groups, seed=271077, draws=10000):
    """지도 평균 차이를 보존하고, 겹치는 지도는 같은 그룹으로 재표집."""
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    if aa.keys() != bb.keys() or not aa:
        raise ValueError('대응되는 시나리오 전체가 일치해야 합니다.')
    regions = sorted({k[0] for k in aa})
    differences = {region:float(np.mean([
        aa[k][metric] - bb[k][metric] for k in aa if k[0] == region
    ])) for region in regions}
    grouped = {}
    for region in regions:
        group = map_groups.get(region)
        if group is None:
            raise ValueError('공간 그룹이 없는 지도: ' + region)
        grouped.setdefault(group, []).append(differences[region])
    groups = sorted(grouped)
    totals = np.array([np.sum(grouped[g]) for g in groups])
    sizes = np.array([len(grouped[g]) for g in groups])
    ix = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    boot = totals[ix].sum(1) / sizes[ix].sum(1)
    return dict(difference=float(np.mean(list(differences.values()))),
                ci95_low=float(np.quantile(boot,.025)), ci95_high=float(np.quantile(boot,.975)),
                regions=len(regions), spatial_groups=len(groups), scenarios=len(aa),
                region_differences=differences,
                method='paired spatial-group percentile bootstrap; map-weighted mean; 10000 draws; descriptive with few independent groups')


def trajectory_audit(arrays, city, summary):
    nodes = arrays['nodes']
    positions = np.asarray(arrays['positions_xy'], float)
    if positions.shape != (len(nodes),8,2) or not np.isfinite(positions).all():
        raise ValueError('유한한 [시점,8,2] 궤적 필요')
    if not np.array_equal(positions, city.points[nodes]):
        raise ValueError('저장 좌표와 지도 노드 불일치')
    a,b = positions[:-1],positions[1:]
    ii,jj=np.triu_indices(8,1)
    r=a[:,ii]-a[:,jj]
    dv=(b[:,ii]-a[:,ii])-(b[:,jj]-a[:,jj])
    dd=(dv*dv).sum(-1)
    t=np.divide(-(r*dv).sum(-1),dd,out=np.zeros_like(dd),where=dd>1e-20)
    minimum=np.sqrt(((r+np.clip(t,0,1)[...,None]*dv)**2).sum(-1))
    collisions=int(np.sum(minimum<5-1e-6))
    segments=shapely.linestrings(np.stack((a,b),axis=-2).reshape(-1,2,2))
    obstacle_segments=int(np.sum(shapely.intersects(segments,city.flight_blocker)))
    obstacle_points=int(np.sum(shapely.intersects_xy(city.flight_blocker,positions[...,0],positions[...,1])))
    bad_edges=int(np.sum((nodes[:-1]!=nodes[1:]) & (np.asarray(city.graph[nodes[:-1].ravel(),nodes[1:].ravel()]).ravel().reshape(nodes[:-1].shape)==0)))
    speed=float(np.linalg.norm(b-a,axis=-1).max()/3)
    statistics=np.asarray(arrays['C'])
    links=np.asarray(arrays['links'],bool)
    if links.shape!=(len(statistics),8,8) or not np.isfinite(statistics).all() or not np.array_equal(links,links.transpose(0,2,1)):
        raise ValueError('유한한 RF 지표와 대칭 링크 기록 필요')
    independently=graph_statistics_from_links(links)
    connected=independently[:,0].astype(bool)
    if statistics.shape!=independently.shape or not np.allclose(statistics,independently,atol=1e-12,rtol=0):
        raise ValueError('원시 링크의 도달 가능성과 연결 지표 불일치')
    if abs(float(connected.mean())-summary['search_connected_mean'])>1e-12:
        raise ValueError('연결 유지율과 원시 기록 불일치')
    down=~connected;begins=np.flatnonzero(down&~np.r_[False,down[:-1]])
    ends=np.flatnonzero(down&~np.r_[down[1:],False])+1
    expected={'outage_total_s':float(down.sum()*.5),
              'outage_max_s':float((ends-begins).max()*.5) if len(begins) else 0.,
              'outage_events':len(begins),'relay_travel_m':float(np.linalg.norm(b-a,axis=-1)[:,3:].sum()),
              'search_travel_m':float(np.linalg.norm(b-a,axis=-1)[:,:3].sum()),
              'pdr':summary['delivered_packets']/max(summary['generated_packets'],1)}
    if any(not np.isclose(summary[k],value,atol=1e-9,rtol=0) for k,value in expected.items()):
        raise ValueError('단절·이동량·전달률의 독립 재계산 불일치')
    if summary['generated_packets']!=sum(summary[k] for k in ['delivered_packets','ttl_drops','retry_drops','overflow_drops']):
        raise ValueError('패킷 보존식 불일치')
    if len(arrays['C'])!=6*(len(nodes)-1) or summary['duration_s']!=3*(len(nodes)-1):
        raise ValueError('궤적·RF·임무 시간 불일치')
    return dict(minimum_swept_separation_m=float(minimum.min()),
                collision_pairs=collisions,obstacle_segments=obstacle_segments,
                obstacle_points=obstacle_points,invalid_graph_edges=bad_edges,
                maximum_speed_mps=speed,packet_conservation=True,metrics_recomputed=True,connectivity_from_links_verified=True,
                passed=collisions==0 and obstacle_segments==0 and obstacle_points==0 and bad_edges==0 and speed<=5+1e-6)


@lru_cache(maxsize=45)
def city_for(region):
    maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    entry=next(e for e in maps if e['key']==region)
    return prepare_city_without_unused_radio_cache(ROOT/entry['path'],Scenario().validate(),ROOT/'cache/usable')


def training_audit():
    rows=[];audits=[];plan=protocol()
    for seed in [27101,27102,27103]:
        initials={}
        for mode in ['risk','no_risk']:
            folder=MODELS/f'{mode}_s{seed}'
            initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
            final=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            if final['seed']!=seed or final['use_risk']!=(mode=='risk') or final['training_sources']!=plan['source_sha256']:
                raise ValueError('독립 학습 체크포인트 계약 불일치')
            initials[mode]=initial['model']
            history=final['history']
            if len(history)!=8 or sum(h['episodes'] for h in history)!=64:
                raise ValueError('완료된 64회 독립 학습 필요')
            largest=max(h['initial_logp_max_abs'] for h in history)
            if largest>2e-5:raise ValueError('PPO 수집 정책 불일치')
            if not all(h['complete_count']==h['episodes'] for h in history):raise ValueError('학습에 미완주 임무 포함')
            actor_change=float(np.sqrt(sum(float((final['model'][k]-initial['model'][k]).double().square().sum()) for k in initial['model'] if k.startswith('actor.'))))
            if actor_change<1e-8:raise ValueError('학습 후 행동망 파라미터 변화 없음')
            paths=sorted((STUDY/'training'/f'{mode}_s{seed}').glob('i[0-9][0-9]_e[0-9][0-9].npz'))
            if len(paths)!=64:raise ValueError('학습 궤적 수 불일치')
            for path in paths:
                meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                if meta.get('refinement_sources')!=plan['source_sha256']:
                    raise ValueError('학습 수집 코드 지문 불일치')
                spec={k:meta[k] for k in ['region','direction','kind','seed']}
                expected=sha256_file(folder/f'iteration_{int(path.stem[1:3]):02d}.pt')
                if validated_result(path.with_suffix(''),spec,expected,rollout=True) is None:
                    raise ValueError('학습 원시 배열·수집 체크포인트 불일치')
                with np.load(path) as z:
                    audit=trajectory_audit(z,city_for(meta['region']),meta)
                if not audit['passed']:raise ValueError('학습 중 실제 궤적 안전 검수 실패')
                audits.append(dict(phase='training',tag=f'{mode}_s{seed}',**spec,**audit))
                if mode=='no_risk':
                    if meta['forecast_calls']!=0 or meta['risk_input_enabled']:
                        raise ValueError('위험 입력 제거 실험에 예측기 실행')
                    with np.load(path) as z:
                        if np.any(z['state'][:,96:160]!=0):raise ValueError('위험 제거 Critic에 예측 입력 누출')
                        for peer in range(7):
                            if np.any(z['context'][...,14+peer*11+6:14+peer*11+9]!=0):
                                raise ValueError('위험 제거 Actor에 예측 입력 누출')
            rows.append(dict(mode=mode,seed=seed,episodes=64,actor_change_l2=actor_change,
                             max_initial_logp_error=largest,all_complete=all(h['complete_count']==h['episodes'] for h in history),
                             checkpoint_sha256=sha256_file(folder/'last.pt')))
        if any(not torch.equal(initials['risk'][k],initials['no_risk'][k]) for k in initials['risk']):
            raise ValueError('대응하는 위험 유무 학습의 초기 가중치가 다름')
    atomic_json(STUDY/'training_trajectory_audit.json',audits)
    return rows


def validation_audit(freeze,plan):
    audits=[]
    for tag,candidate in freeze['candidates'].items():
        file=STUDY/'validation'/f'{tag}_complete.json'
        rows=json.loads(file.read_text(encoding='utf-8'))['results']
        expected=[s for s in plan['validation'] if candidate['category']!='seed_diagnostic' or s['direction']=='east_west']
        if len(rows)!=len(expected) or {key(r) for r in rows}!={key(s) for s in expected}:
            raise ValueError('사전 지정한 검증·난수 진단 조건 불일치')
        for row in rows:
            stem=f'{row["region"]}_{row["direction"]}_{row["kind"]}_{row["seed"]}'
            prefix=STUDY/'validation'/tag/stem
            spec={k:row[k] for k in ['region','direction','kind','seed']}
            if (validated_result(prefix,spec,freeze['checkpoint_sha256'].get(tag)) is None
                    or row.get('refinement_sources')!=plan['source_sha256']):
                raise ValueError('검증 원시 파일 무결성·코드 지문 실패')
            with np.load(prefix.with_suffix('.npz')) as z:audit=trajectory_audit(z,city_for(row['region']),row)
            if not audit['passed']:raise ValueError('검증 중 실제 궤적 안전 검수 실패')
            audits.append(dict(phase='validation',tag=tag,**spec,**audit))
    if len(audits)!=336:raise ValueError('336회 검증·진단 전체 필요')
    atomic_json(STUDY/'validation_trajectory_audit.json',audits)
    return audits


def safety_summary(audits):
    return dict(episodes_checked=len(audits),all_passed=all(a['passed'] for a in audits),
                minimum_swept_separation_m=min(a['minimum_swept_separation_m'] for a in audits),
                collision_pairs=sum(a['collision_pairs'] for a in audits),
                obstacle_segments=sum(a['obstacle_segments'] for a in audits),
                invalid_graph_edges=sum(a['invalid_graph_edges'] for a in audits))


def analyze():
    plan=protocol()
    from cooperative.contracts import verify_forecaster_files
    forecaster_files=verify_forecaster_files()
    freeze=json.loads((STUDY/'selection.json').read_text(encoding='utf-8'))
    protocol_maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    map_groups={e['key']:e['spatial_group'] for e in protocol_maps}
    expected={tuple(s[k] for k in ['region','direction','kind','seed']) for s in freeze['test_specs']}
    all_rows=[];groups={};audits=[]
    for file in sorted((STUDY/'test').glob('*_complete.json')):
        data=json.loads(file.read_text(encoding='utf-8'));tag=file.stem[:-len('_complete')]
        items=sorted(data['results'],key=key)
        if {key(r) for r in items}!=expected or len(items)!=32:
            raise ValueError('사전에 고정한 시험 32조건과 불일치: '+tag)
        for result in items:
            stem=f'{result["region"]}_{result["direction"]}_{result["kind"]}_{result["seed"]}'
            prefix=STUDY/'test'/tag/stem
            spec={k:result[k] for k in ['region','direction','kind','seed']}
            if validated_result(prefix,spec,result.get('checkpoint_sha256')) is None:
                raise ValueError('원시 시험 파일 무결성 실패')
            if result.get('refinement_sources')!=plan['source_sha256']:
                raise ValueError('시험 코드 지문 불일치')
            if result.get('checkpoint_sha256')!=freeze['checkpoint_sha256'].get(tag):
                raise ValueError('검증 이후 시험 체크포인트 변경')
            expected_risk=tag not in ['nominal','reactive','mpc_no_risk','no_risk_ensemble','no_risk_ensemble_guarded']
            if result['risk_input_enabled']!=expected_risk or (not expected_risk and result['forecast_calls']!=0):
                raise ValueError('시험 예측 위험 입력 계약 불일치')
            with np.load(prefix.with_suffix('.npz')) as arrays:
                audit=trajectory_audit(arrays,city_for(result['region']),result)
                result.update({name:float(np.asarray(arrays['C'])[:,column].mean()) for name,column in NETWORK_COLUMNS.items()})
            if not audit['passed']:raise ValueError('독립 궤적 안전 검수 실패')
            audits.append(dict(phase='test',tag=tag,region=result['region'],seed=result['seed'],direction=result['direction'],kind=result['kind'],**audit))
            all_rows.append(dict(method=tag,spatial_group=map_groups[result['region']],**{k:v for k,v in result.items() if isinstance(v,(str,int,float,bool)) or v is None}))
        groups[tag]=items
    methods={'nominal','reactive','mpc','mpc_no_risk','legacy_ppo','risk_ensemble','risk_ensemble_guarded','no_risk_ensemble','no_risk_ensemble_guarded'}
    if set(groups)!=methods:raise ValueError('사전 고정한 비교 방법 9개 필요')
    summaries={tag:summarize(items) for tag,items in groups.items()}
    metrics=['search_connected_mean','pdr','outage_total_s','outage_max_s','duration_ratio','relay_travel_m','timely_pdr']
    vs_nominal={tag:{metric:paired_spatial_interval(items,groups['nominal'],metric,map_groups) for metric in metrics} for tag,items in groups.items() if tag!='nominal'}
    matched={}
    for suffix in ['', '_guarded']:
        matched['risk_vs_no_risk'+suffix]={metric:paired_spatial_interval(groups['risk_ensemble'+suffix],groups['no_risk_ensemble'+suffix],metric,map_groups) for metric in metrics}
    matched['mpc_risk_vs_no_risk']={metric:paired_spatial_interval(groups['mpc'],groups['mpc_no_risk'],metric,map_groups) for metric in metrics}
    fields=['method','spatial_group','region','direction','kind','seed','mission_completed',*METRICS,*NETWORK_COLUMNS,'generated_packets','delivered_packets','ttl_drops','retry_drops','overflow_drops','beacon_tx','beacon_rx','forecast_calls','risk_input_enabled','checkpoint_sha256']
    fields=list(dict.fromkeys(fields))
    with (STUDY/'comparison_all_metrics_v2.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    training=training_audit();validation=validation_audit(freeze,plan)
    train_audits=json.loads((STUDY/'training_trajectory_audit.json').read_text(encoding='utf-8'))
    formal=train_audits+validation+audits
    if len(formal)!=1008:raise ValueError('1,008회 공식 학습·검증·시험 검수 필요')
    result=dict(format='refinement-audited-results-v2',methods=len(groups),test_scenarios_per_method=32,test_episodes=len(all_rows),
                regions=8,independent_spatial_groups=6,selected_by_validation=freeze['selected'],
                selection_sha256=sha256_file(STUDY/'selection.json'),source_sha256=plan['source_sha256'],forecaster_files_sha256=forecaster_files,
                completed_test_sha256={file.name:sha256_file(file) for file in sorted((STUDY/'test').glob('*_complete.json'))},
                summaries=summaries,vs_nominal=vs_nominal,matched_forecast_ablation=matched,
                training_audit=training,safety=safety_summary(audits),formal_safety=safety_summary(formal))
    atomic_json(STUDY/'trajectory_audit.json',audits)
    atomic_json(STUDY/'all_formal_trajectory_audit.json',formal)
    atomic_json(STUDY/'audited_summary.json',result)
    print(json.dumps({'audited_episodes':len(audits),'selected':freeze['selected'],'safety':result['safety']},ensure_ascii=False))
    return result


if __name__=='__main__':analyze()
