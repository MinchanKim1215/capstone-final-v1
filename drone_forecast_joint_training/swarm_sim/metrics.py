"""지표를 각각 계산한다. 서로 다른 단위의 지표를 가중합하지 않는다."""
# 초기 엔진의 지표 계산과 파레토 비교를 담은 파일입니다.
# 현재 전체 임무의 실제 지표 집계는 full_run.py에 있으므로 두 엔진의 키·분모를 구분합니다.
# 파레토는 다른 지표를 더해 총점을 만드는 방식이 아니라, 모든 축에서 뒤처지는 설계를 가리는 방식입니다.
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from .network import jain


# 초기 그래프의 수색연결·전체연결·쌍도달·성분·차수·효율·정규화λ₂를 각각 계산합니다.
def graph_metrics(adj,n_search):
    n=len(adj);graph=csr_matrix(adj.astype(float))
    nc,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels)
    ii,jj=np.triu_indices(n_search,1)
    reach=labels[ii]==labels[jj]
    dist=shortest_path(graph,directed=False,unweighted=True)
    hops=dist[ii,jj];finite=np.isfinite(hops)
    all_i,all_j=np.triu_indices(n,1)
    efficiency=np.where(np.isfinite(dist[all_i,all_j]),1/np.maximum(dist[all_i,all_j],1),0).mean()
    degree=adj.sum(axis=1).astype(float)
    inv=np.where(degree>0,1/np.sqrt(np.maximum(degree,1)),0)
    lap=np.diag(degree>0)-inv[:,None]*adj*inv[None,:]
    eig=np.linalg.eigvalsh(lap)
    return {"fleet_connected":float(nc==1),"search_connected":float(reach.all()),
            "search_pair_reachability":float(reach.mean()),"components":int(nc),
            "largest_component_fraction":float(sizes.max()/n),"mean_degree":float(degree.mean()),
            "isolated_fraction":float(np.mean(degree==0)),"edge_density":float(adj[all_i,all_j].mean()),
            "normalized_lambda2":float(max(0,eig[1])),"graph_efficiency":float(efficiency),
            "reachable_search_hops":float(hops[finite].mean()) if finite.any() else None}


# 한 노드를 가상 제거한 정적 그래프에서 남은 수색 드론이 같은 성분인지 확인합니다.
def search_survives_removal(adj,n_search,node):
    remaining=[i for i in range(len(adj)) if i!=node]
    search=[remaining.index(i) for i in range(n_search) if i!=node]
    if len(search)<2:return True
    _,labels=connected_components(csr_matrix(adj[np.ix_(remaining,remaining)]),directed=False)
    return len(set(labels[search]))==1


# 명목 간격이 아닌 실제 비행 위치의 최근접 거리·최대거리·수색군 반경을 모읍니다.
class SpatialDistances:
    """명목 대형 간격과 실제 비행 중 거리를 구분한다."""
    # 거리 표본 목록을 초기화합니다. 초기 엔진의 대형·역할 개수와 연결됩니다.
    def __init__(self,n_search):
        self.n_search=n_search;self.nearest=[];self.search_nearest=[]
        self.maximum=[];self.search_pairs=[];self.search_radius=[]

    # 활성 드론과 수색 드론의 실제 쌍 거리를 프레임별로 모읍니다.
    def push(self,positions,active):
        p=positions[active];dist=np.linalg.norm(p[:,None]-p[None,:],axis=2)
        np.fill_diagonal(dist,np.inf)
        if len(p)>1:
            self.nearest.extend(np.min(dist,axis=1).tolist())
            self.maximum.append(float(np.max(dist[np.isfinite(dist)])))
        search=positions[:self.n_search]
        sd=np.linalg.norm(search[:,None]-search[None,:],axis=2);np.fill_diagonal(sd,np.inf)
        self.search_nearest.append(float(np.min(sd,axis=1).mean()))
        self.search_pairs.append(float(sd[np.triu_indices(len(search),1)].mean()))
        self.search_radius.append(float(np.max(np.linalg.norm(search-search.mean(axis=0),axis=1))))

    # 모은 실제 거리의 평균·5백분위와 수색 배치 반경을 반환합니다.
    def finalize(self):
        return {"nearest_neighbor_distance_mean_m":float(np.mean(self.nearest)),
                "nearest_neighbor_distance_p05_m":float(np.percentile(self.nearest,5)),
                "maximum_pair_distance_mean_m":float(np.mean(self.maximum)),
                "search_nearest_neighbor_mean_m":float(np.mean(self.search_nearest)),
                "search_pair_distance_mean_m":float(np.mean(self.search_pairs)),
                "search_spread_radius_mean_m":float(np.mean(self.search_radius))}


# 초기 엔진의 연결·관측·공유·대형오차·안전거리·이동량 집계입니다.
class EpisodeMetrics:
    # 시각별 지표 목록과 링크 변화·최소 안전거리 상태를 초기화합니다.
    def __init__(self,design,scenario,city):
        self.d,self.s,self.city=design,scenario,city
        self.rows=[];self.previous_adj=None;self.link_changes=0
        self.minimum_separation=float("inf")
        self.spatial=SpatialDistances(design.n_search)

    # 현재 프레임의 그래프·센서·지식·이동을 지표로 바꿉니다. 결과 지표를 예측 입력으로 쓰면 시각 정렬에 주의해야 합니다.
    def push(self,t,state,motion,old,new,desired,observed,known,network,jammer_on):
        d=self.d;ns=d.n_search;n=d.n_drones
        row={"t":round(t,6),**graph_metrics(state["adjacency"],ns)}
        from .motion import swept_minimum
        self.minimum_separation=min(self.minimum_separation,swept_minimum(old,new,motion.active)[0])
        self.spatial.push(old,motion.active)
        each=np.array([len(x) for x in observed]); union=set().union(*observed)
        shared=set.intersection(*known[:ns]) if ns else set()
        den=len(self.city.target_points)
        err=(old[:ns]-old[:ns].mean(axis=0))-(desired[:ns]-desired[:ns].mean(axis=0))
        ii,jj=np.triu_indices(n,1)
        survival=np.mean([search_survives_removal(state["adjacency"],ns,i) for i in range(n)])
        relay_survival=np.mean([search_survives_removal(state["adjacency"],ns,i) for i in range(ns,n)]) if d.n_relays else None
        row.update({"observed_coverage":len(union)/den,"shared_coverage":len(shared)/den,
                    "coverage_jain":jain(each),"coverage_overlap":1-len(union)/each.sum() if each.sum() else 0,
                    "formation_rmse_m":float(np.sqrt(np.mean(np.sum(err*err,axis=1)))),
                    "los_fraction":float(state["los"][ii,jj].mean()),
                    "sinr_p05_db":float(np.percentile(np.minimum(state["sinr_db"][ii,jj],state["sinr_db"][jj,ii]),5)),
                    "sinr_mean_db":float(np.mean(state["sinr_db"][ii,jj])),
                    "rx_mean_dbm":float(np.mean(state["rx_dbm"][ii,jj])),
                    "path_loss_mean_db":float(np.mean(state["loss_db"][ii,jj])),
                    "single_node_removal_survival":float(survival),"relay_removal_survival":float(relay_survival) if relay_survival is not None else None,
                    "pdr_so_far":network.delivered/network.generated if network.generated else 0,
                    "queue_packets":sum(map(len,network.queues)),"jammer_on":bool(jammer_on)})
        if self.previous_adj is not None:self.link_changes+=int(np.count_nonzero(np.triu(self.previous_adj!=state["adjacency"],1)))
        self.previous_adj=state["adjacency"].copy();self.rows.append(row)

    # 초기 실험 구간의 평균·단절시간·수색 공유와 실제 이동량을 반환합니다. 에너지J로 환산하지 않습니다.
    def finalize(self,motion):
        rows=self.rows;result={}
        if not rows:raise RuntimeError("비어 있는 실험입니다.")
        for key in ("fleet_connected","search_connected","search_pair_reachability","components",
                    "largest_component_fraction","mean_degree","isolated_fraction","edge_density",
                    "normalized_lambda2","graph_efficiency","reachable_search_hops","formation_rmse_m",
                    "los_fraction","sinr_p05_db","sinr_mean_db","rx_mean_dbm","path_loss_mean_db",
                    "single_node_removal_survival","relay_removal_survival","queue_packets"):
            a=np.array([r[key] for r in rows if r[key] is not None],float)
            result[key+"_mean"]=float(a.mean()) if len(a) else None
        flags=np.array([r["search_connected"]==0 for r in rows]);runs=[];length=0
        for fail in flags:
            if fail:length+=1
            elif length:runs.append(length*self.s.motion_dt_s);length=0
        if length:runs.append(length*self.s.motion_dt_s)
        result.update({"search_outage_total_s":float(flags.sum()*self.s.motion_dt_s),
                       "search_outage_events":len(runs),"search_outage_max_s":max(runs,default=0),
                       "formation_rmse_p95_m":float(np.percentile([r["formation_rmse_m"] for r in rows],95)),
                       "observed_coverage_final":rows[-1]["observed_coverage"],
                       "shared_coverage_final":rows[-1]["shared_coverage"],
                       "coverage_jain_final":rows[-1]["coverage_jain"],
                       "coverage_overlap_final":rows[-1]["coverage_overlap"],
                       "total_travel_m":float(motion.distance.sum()),
                       "search_travel_m":float(motion.distance[:self.d.n_search].sum()),
                       "relay_travel_m":float(motion.distance[self.d.n_search:].sum()),
                       "minimum_swept_separation_m":self.minimum_separation,
                       "stationary_agent_frames":motion.wait_count,"boundary_clamp_agent_frames":motion.boundary_clamps,
                       "link_changes":self.link_changes,"link_changes_per_s":self.link_changes/self.s.duration_s,
                       "obstacle_violations":0,"flight_free_fraction":self.city.flight_free_fraction,
                       "retained_free_cell_fraction":self.city.retained_free_cell_fraction,
                       "jammer_active_fraction":float(np.mean([r["jammer_on"] for r in rows]))})
        # 센서·패킷 처리 이후 지표이므로 현재 시점 예측 입력으로 그대로 쓰지 않는다.
        return {**result,**self.spatial.finalize()}


# 다른 설계가 모든 축에서 같거나 좋고 한 축 이상 더 좋으면 해당 설계를 지배합니다.
# 서로 다른 단위 지표를 합산한 종합 점수는 만들지 않습니다.
def pareto_mask(values,maximize):
    a=np.asarray(values,float)*np.where(np.asarray(maximize),1,-1)
    valid=np.isfinite(a).all(axis=1);mask=valid.copy()
    for i in np.flatnonzero(valid):
        dominates=np.all(a>=a[i]-1e-12,axis=1)&np.any(a>a[i]+1e-12,axis=1)&valid
        if dominates.any():mask[i]=False
    return mask
