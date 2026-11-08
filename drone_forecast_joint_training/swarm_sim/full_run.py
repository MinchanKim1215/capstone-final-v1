"""완전한 이동 임무, 건물 기반 채널, 추상 패킷을 결합한 실행 경로."""
# 현재 실험의 통합 실행 파일입니다.
# 먼저 통신과 무관한 이동 궤적을 계산하고, 그 위에서 채널·패킷·성능 지표를 계산합니다.
# 따라서 이 버전에서는 통신이 나빠져도 비행 경로가 통신 위험 때문에 바뀌지는 않습니다.
# 이동·센서는 3초, 채널·라우팅은 전수 실험 3초 / 후보 재검증 0.5초입니다.
# 그래프 연결, 실제 목적지 패킷 전달, 수색 관측, 정보 공유는 각각 별도 지표입니다.
from dataclasses import asdict
import numpy as np
import shapely
from .mission import full_mission,MOVE_DT
from .fast_network import probabilities,packet_run,graph_statistics
from .motion import fleet_offsets

# 정규화 라플라시안의 두 번째 고유값과 노드 한 대 제거 후 수색군 연결 여부를 계산합니다.
# 정적 그래프 강건성 진단이며 실제 드론 이탈 사건이나 복구 알고리즘 실험이 아닙니다.
def spectral_and_survival(adjacency,ns):
    n=len(adjacency);degree=adjacency.sum(axis=1)
    inv=np.zeros(n);inv[degree>0]=1/np.sqrt(degree[degree>0])
    lap=np.diag(degree>0)-adjacency*inv[:,None]*inv[None,:]
    lam=float(np.linalg.eigvalsh(lap)[1])
    survive=[]
    for removed in range(n):
        keep=[i for i in range(n) if i!=removed];search=[i for i in range(ns) if i!=removed]
        if len(search)<2:survive.append(1.);continue
        visited={search[0]};pending=[search[0]]
        while pending:
            i=pending.pop()
            for j in keep:
                if adjacency[i,j] and j not in visited:visited.add(j);pending.append(j)
        survive.append(float(all(i in visited for i in search)))
    return lam,float(np.mean(survive)),float(np.mean(survive[ns:])) if ns<n else None

# 한 지도·한 배치·한 난수·간섭 조건의 전체 임무를 계산합니다.
# 이동 궤적을 공유하면 간섭 ON/OFF에서 비행을 바꾸지 않고 통신 효과를 비교할 수 있습니다.
# capture=True일 때만 화면용 좌표·궤적·관측 셀·지표 프레임을 함께 저장합니다.
def run_full(city,design,scenario,seed,jammer_on=True,radio_dt=3.,capture=False,mission=None,disable_relay_radios=False):
    # 빠른엔진안에고정상수가있으므로설정파일만바꿔다른모델인척실행하지못하게합니다.
    if not (scenario.mac=='tdma' and scenario.radio_model=='bullington' and scenario.routing=='expected_attempts'
       and scenario.frequency_hz==2.4e9 and scenario.tx_power_dbm==10 and scenario.gain_dbi==0
       and scenario.bandwidth_hz==1e6 and scenario.noise_figure_db==7 and scenario.bit_rate_bps==1e6
       and scenario.packet_payload_bytes==512 and scenario.packet_header_bytes==64 and scenario.ack_bytes==14
       and scenario.slot_s==.005 and scenario.total_packets_per_s==30 and scenario.queue_limit==64
       and scenario.max_hop_attempts==4 and scenario.packet_lifetime_s==2 and scenario.timely_deadline_s==.25
       and scenario.link_success_threshold==.9 and scenario.traffic_mode=='fixed_total'):
        raise ValueError('고속 전체 임무 엔진의 고정 채널·MAC·패킷 설정을 변경하려면 엔진도 수정해야 합니다.')
    if scenario.motion_dt_s!=3 or scenario.min_separation_m!=5 or scenario.relay_failure_fraction is not None:
        raise ValueError('전체 임무는 이동3초·안전5m·이탈 없음 조건입니다.')
    if radio_dt not in (3.,.5):raise ValueError('radio_dt는 3 또는 0.5초')
    if mission is None:mission=full_mission(city,design,scenario)
    ratio=int(round(MOVE_DT/radio_dt));coarse=mission['nodes'];n=design.n_drones;ns=design.n_search
    # 3초이동궤적을radio_dt주기에선형보간합니다. 0.5초재검증도새로운비행정책을쓰는것은아닙니다.
    k=np.repeat(np.arange(len(coarse)-1),ratio);alpha=np.tile(np.arange(ratio)/ratio,len(coarse)-1)
    nodes=coarse[k];xyz0=city.points[coarse[k]];xyz1=city.points[coarse[k+1]]
    xy=xyz0*(1-alpha[:,None,None])+xyz1*alpha[:,None,None]
    xyz=np.concatenate([xy,np.full((*xy.shape[:-1],1),scenario.altitude_m)],axis=2)
    nt=len(xy);t=np.arange(nt)*radio_dt;total=mission['duration_s']
    # 간섭원위치를seed로정합니다. 전파섀도잉seed와패킷성공seed(+1000003)를별도로사용합니다.
    rng=np.random.default_rng(seed);jcell=int(rng.integers(len(city.target_points)))
    jammer=np.r_[city.target_points[jcell],scenario.jammer_height_m]
    active=(t>=scenario.jammer_start_fraction*total)&(t<scenario.jammer_end_fraction*total)&jammer_on
    if ratio==1:
        loss=city.loss_cache[nodes[:,:,None],nodes[:,None,:]].astype(np.float64)
        los=city.los_cache[nodes[:,:,None],nodes[:,None,:]]
    else:
        loss=np.empty((nt,n,n));los=np.empty((nt,n,n),bool)
        for st in range(0,nt,100):
            pp=xyz[st:st+100];tx=np.repeat(pp,n,axis=1).reshape(-1,3);rx=np.tile(pp,(1,n,1)).reshape(-1,3)
            a,b,_=city.radio.propagation(tx,rx,scenario)
            loss[st:st+len(pp)]=a.reshape(len(pp),n,n);los[st:st+len(pp)]=b.reshape(len(pp),n,n)
    jloss,_,_=city.radio.propagation(jammer[None,:],xyz.reshape(-1,3),scenario)
    jmw=10.**((scenario.jammer_power_dbm-jloss.reshape(nt,n))/10)
    interference=jmw*active[:,None]
    prob=probabilities(loss,interference,radio_dt,seed,scenario.shadow_sigma_db,scenario.shadow_correlation_s)
    # packet_run반환배열에서실제목적지수신을집계합니다. 그래프연결만으로전달성공을가정하지않습니다.
    values,rows,graphs,status,delays,hops,relayhops,observed,known,psrc,pdst=packet_run(prob,nodes,city.visible,ns,radio_dt,seed+1000003,disable_relay_radios)
    generated,delivered,attempts,successful,retries,expired,retrydrop,overflow,timely,relay_success,aoi,outage,outagemax,outevents=values
    # 모든생성패킷의운명을보존식으로확인합니다. 일부패킷을분모에서제거해PDR을높이지않습니다.
    if int(generated)!=int(delivered+expired+retrydrop+overflow):raise RuntimeError('패킷 보존식 위반')
    good=status==1;obsunion=np.any(observed,axis=0);intersection=np.all(known,axis=0)
    obsper=observed.sum(axis=1);fair=float(obsper.sum()**2/(ns*np.sum(obsper**2))) if np.any(obsper) else 0.
    flow_pdr=[];flow_min=[]
    for i in range(ns):
        for j in range(ns):
            if i!=j:
                mask=(psrc==i)&(pdst==j);flow_pdr.append(float(np.sum(good&mask)/max(1,np.sum(mask))))
    f=np.asarray(flow_pdr)
    grkeys=['search_connected_mean','search_pair_reachability_mean','fleet_connected_mean','components_mean',
      'largest_component_fraction_mean','mean_degree_mean','isolated_fraction_mean','edge_density_mean',
      'graph_efficiency_mean','reachable_search_hops_mean']
    m={key:float(value) for key,value in zip(grkeys,graphs.mean(axis=0))}
    m.update({key:value for key,value in mission.items() if key not in ('nodes','desired','phase')})
    observable=int(city.observable.sum());targets=len(city.target_points)
    # PDR은종단간전달/생성, 지연은전달성공패킷에조건부입니다. 종료후2초정산수신도포함됩니다.
    # goodput은이생성코호트의전달payload를비행시간으로나눈값이며엄밀한시간창처리량과구분합니다.
    m.update({'generated':int(generated),'delivered':int(delivered),'pdr':float(delivered/generated),
      'timely_pdr':float(timely/generated),'goodput_kbps':float(delivered*512*8/total/1000),
      'delay_mean_s':float(np.mean(delays[good])) if good.any() else None,
      'delay_p95_s':float(np.quantile(delays[good],.95)) if good.any() else None,
      'delay_max_s':float(np.max(delays[good])) if good.any() else None,
      'hops_mean':float(hops[good].mean()) if good.any() else None,'hops_p95':float(np.quantile(hops[good],.95)) if good.any() else None,
      'tx_attempts':int(attempts),'hop_successes':int(successful),'retries':int(retries),
      'drop_expired':int(expired),'drop_retry_limit':int(retrydrop),'drop_queue_overflow':int(overflow),
      'hop_delivery_ratio':float(successful/max(attempts,1)),
      'transmission_attempts_per_delivered':float(attempts/max(delivered,1)),
      'relay_successful_hops':int(relay_success),'relay_hop_fraction':float(relay_success/max(successful,1)),
      'flow_pdr_min':float(f.min()),'flow_pdr_jain':float(f.sum()**2/(len(f)*np.sum(f*f))) if np.any(f) else 0.,
      'aoi_mean_s':float(aoi),'queue_packets_mean':float(rows[:,5].mean()),
      'search_outage_total_s':float(outage),'search_outage_max_s':float(outagemax),'search_outage_events':int(outevents),
      'target_cells':targets,'observable_cells':observable,'unobservable_cells':targets-observable,
      'observed_cells_final':int(obsunion.sum()),'observed_coverage_final':float(obsunion.mean()),
      'observable_coverage_final':float(np.sum(obsunion&city.observable)/observable),
      'shared_coverage_final':float(intersection.mean()),'shared_observable_coverage_final':float(np.sum(intersection&city.observable)/observable),
      'coverage_jain_final':fair,'coverage_overlap_final':float(1-obsunion.sum()/max(obsper.sum(),1)),
      'wire_attempt_kbytes':float(attempts*(512+64+14)/1000),'mission_success':bool(mission['mission_completed'] and intersection.mean()>=.9),
      'disabled_relay_radios':bool(disable_relay_radios)})
    off=fleet_offsets(design,scenario.direction)[:ns];p=xy[:,:ns]
    error=np.linalg.norm((p-p.mean(axis=1)[:,None,:])-off[None,:,:],axis=2)
    m['formation_rmse_m_mean']=float(np.sqrt(np.mean(error**2,axis=1)).mean())
    # 이키의95백분위는모든시간×개별수색드론오차분포입니다. 프레임별RMS의95백분위가아닙니다.
    m['formation_rmse_p95_m']=float(np.quantile(error,.95))
    d=np.linalg.norm(xy[:,:,None,:]-xy[:,None,:,:],axis=3);d[:,np.arange(n),np.arange(n)]=np.inf
    m['nearest_neighbor_distance_mean_m']=float(d.min(axis=2).mean())
    upper=np.triu_indices(n,1);m['los_fraction_mean']=float(los[:,upper[0],upper[1]].mean())
    m['path_loss_mean_db_mean']=float(loss[:,upper[0],upper[1]].mean())
    # 여기보조SINR·수신전력통계는섀도잉을뺀기하값입니다. 실제패킷확률은섀도잉포함이므로정의를혼용하지않습니다.
    den=-107+10*np.log10(1+interference/10**(-107/10))
    sinr=10-loss-den[:,None,:]
    m['sinr_mean_db_mean']=float(sinr[:,upper[0],upper[1]].mean())
    m['sinr_p05_db_mean']=float(np.quantile(np.minimum(sinr,sinr.transpose(0,2,1))[:,upper[0],upper[1]],.05,axis=1).mean())
    m['rx_mean_dbm_mean']=float((10-loss[:,upper[0],upper[1]]).mean())
    # Above radio descriptors exclude random shadow term, stated in catalog.
    adjacency=(prob>=.9)&(prob.transpose(0,2,1)>=.9)
    if disable_relay_radios:adjacency[:,ns:,:]=False;adjacency[:,:,ns:]=False
    # 같은인접그래프를묶어고유값·가상노드제거결과를재사용합니다. 지표정의는동일합니다.
    bits=np.packbits(adjacency.reshape(nt,-1),axis=1)
    unique,inv=np.unique(bits,axis=0,return_inverse=True);spec=[]
    for bb in unique:
        a=np.unpackbits(bb)[:n*n].reshape(n,n).astype(bool);spec.append(spectral_and_survival(a,ns))
    for col,key in enumerate(['normalized_lambda2_mean','single_node_removal_survival_mean','relay_removal_survival_mean']):
        m[key]=None if col==2 and ns==n else float(np.asarray([x[col] for x in spec],float)[inv].mean())
    m['link_changes']=int(np.sum(adjacency[1:]!=adjacency[:-1])/2)
    result={'region':city.key,'design':asdict(design),'scenario':asdict(scenario),'seed':int(seed),
      'jammer_on':bool(jammer_on),'radio_dt_s':radio_dt,'metrics':m,'map':city.summary(),'jammer_xyz':jammer.tolist()}
    # 재생용시각정렬주의:좌표는구간시작,누적패킷지표는그표시구간끝값입니다.
    # 이프레임을예측학습입력으로그대로쓰면미래정보가섞일수있으므로학습로그를별도정렬해야합니다.
    if capture:
        frames=[]
        for ti in range(0,nt,ratio):
            row=rows[ti+ratio-1];idx=ti//ratio;edges=np.transpose(np.where(np.triu(adjacency[ti],1)))
            observed_here=[]
            # Captured observations accumulate from the actual 3s sensor nodes.
            if idx==0:obs=np.zeros(targets,bool)
            obs|=np.any(city.visible[coarse[idx,:ns]],axis=0);observed_here=np.flatnonzero(obs).tolist()
            frames.append({'t':float(t[ti]),'positions':xyz[ti].tolist(),'edges':edges.tolist(),
              'observed_cells':observed_here,'phase':int(mission['phase'][idx]),'metrics':{
              'search_connected':bool(graphs[ti,0]),'search_pair_reachability':float(graphs[ti,1]),
              'pdr_so_far':float(row[2]),'observed_coverage':float(row[3]),'shared_coverage':float(row[4]),
              'queue_packets':int(row[5]),'jammer_on':bool(active[ti]),'formation_rmse_m':float(np.sqrt(np.mean(error[ti]**2))),
              'sinr_p05_db':float(np.quantile(sinr[ti,upper[0],upper[1]],.05))}})
        final=frames[-1].copy();final['t']=float(total);final['positions']=np.column_stack([city.points[coarse[-1]],np.full(n,scenario.altitude_m)]).tolist()
        final['metrics']=dict(final['metrics'],pdr_so_far=m['pdr'],observed_coverage=m['observed_coverage_final'],shared_coverage=m['shared_coverage_final'])
        final['observed_cells']=np.flatnonzero(obsunion).tolist();frames.append(final)
        result.update({'frames':frames,'target_points':city.target_points.tolist(),'effective_heights_m':city.heights.tolist(),
          'no_fly_geometry':shapely.to_geojson(city.flight_blocker),'display_dt_s':MOVE_DT,'route':city.route.tolist()})
        result['no_fly_geometry']=__import__('json').loads(result['no_fly_geometry'])
    return result
