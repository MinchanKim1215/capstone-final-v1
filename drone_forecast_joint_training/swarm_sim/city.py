"""지도 로더와 충돌 없는 격자 최단 경로. 경로 계획은 통신 재배치 정책과 구분한다."""
# 지도에서 안전한 비행 그래프와 실외 수색 셀을 만드는 파일입니다.
# 건물 외곽선·높이는 통신 계산에, 50m 이상 건물의 안전 버퍼는 비행 제한에 씁니다.
# 지도 좌표는 동(x)·북(y) 방향 미터이며 지면은 0m로 가정합니다.
# 폴리곤 안을 통과하지 않는 선분만 그래프 간선으로 허용합니다.
from pathlib import Path
import hashlib
import json
import numpy as np
import shapely
from shapely.geometry import shape, box
from shapely.ops import unary_union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from .radio import BuildingRadio

# 건물 지도·안전 영역·비행 노드·경로·센서와 채널 원모델을 보관합니다.
class City:
    # 지도 유효성 확인 → 비행 장애물 생성 → 안전 간선 그래프 → 최대 연결 성분 → 최단 경로 순서입니다.
    # 고립된 작은 비행 성분은 제외하되 제외 비율을 summary에 공개합니다. 목표 실외 셀의 분모는 별도로 유지합니다.
    def __init__(self, path, scenario):
        self.file_path = Path(path)
        raw = self.file_path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        scene = json.loads(raw)
        self.metadata = scene["metadata"]
        self.key = self.metadata["region_key"]
        self.name = self.metadata["region_name"]
        self.bounds = list(self.metadata["local_bounds"])
        if self.bounds != [-200, -200, 200, 200]:
            raise ValueError("현재 기준 구현은 중심 원점의 400m 정사각형 지도입니다.")
        self.buildings = scene["buildings"]
        ids = [b["building_id"] for b in self.buildings]
        if len(set(ids)) != len(ids):
            raise ValueError("지도에 중복 건물 ID가 있습니다.")
        self.geometries = [shape(b["geometry_local_m"]) for b in self.buildings]
        if any(g.is_empty or not g.is_valid for g in self.geometries):
            raise ValueError("지도 건물 기하가 비어 있거나 유효하지 않습니다.")
        self.heights = np.array([b["height_m"] for b in self.buildings], float)
        if not np.isfinite(self.heights).all() or np.any(self.heights <= 0):
            raise ValueError("미확인·0·음수 높이를 자유공간으로 처리하지 않습니다.")
        # imputed는실측/확인높이가아닌추정높이표시입니다. 현재전체실험은전달된추정값그대로사용합니다.
        self.imputed = np.array([b["height_source"] == "imputed" for b in self.buildings])
        shift = {"delivered":0, "imputed_minus15":-15, "imputed_plus15":15}[scenario.height_case]
        self.heights[self.imputed] = np.maximum(1.0, self.heights[self.imputed] + shift)
        self.footprint = unary_union(self.geometries)
        self.aoi = box(*self.bounds)
        policy = self.metadata.get("flight_policy", {})
        reuse = (shift == 0 and scenario.vertical_clearance_m == 0
                 and policy.get("flight_altitude_m") == scenario.altitude_m
                 and policy.get("safety_margin_m") == scenario.horizontal_clearance_m)
        if reuse:
            zones = [shape(z["geometry_local_m"]) for z in scene["no_fly_zones"]]
        else:
            zones = [g.buffer(scenario.horizontal_clearance_m).intersection(self.aoi)
                     for g,h in zip(self.geometries, self.heights)
                     if h + scenario.vertical_clearance_m >= scenario.altitude_m]
        # 비행용버퍼와무선용원래건물폴리곤을분리합니다. 안전버퍼를전파장애물로쓰지않습니다.
        self.flight_blocker = unary_union(zones)
        self.radio = BuildingRadio(self.geometries, self.heights)
        g = scenario.grid_m
        axis = np.arange(-200+g/2, 200, g)
        if not np.isclose(400/g, round(400/g)):
            raise ValueError("격자 간격은 400m를 정수 개로 나눌 수 있어야 합니다.")
        all_points = np.array([(x,y) for y in axis for x in axis])
        self.target_points = all_points[~shapely.intersects_xy(self.footprint, all_points[:,0], all_points[:,1])]
        if not len(self.target_points):
            raise ValueError("수색 대상인 실외 지면 셀이 없습니다.")
        free = ~shapely.intersects_xy(self.flight_blocker, all_points[:,0], all_points[:,1])
        ids = np.flatnonzero(free); points = all_points[ids]
        lookup = {int(index):i for i,index in enumerate(ids)}; width = len(axis)
        pairs = []
        for i,index in enumerate(ids):
            row,col = divmod(int(index),width)
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                rr,cc = row+dr,col+dc
                if 0 <= rr < width and 0 <= cc < width:
                    other = lookup.get(rr*width+cc)
                    if other is not None:
                        pairs.append((i,other))
        if not pairs:
            raise ValueError("비행 가능한 격자 경로가 없습니다.")
        pairs = np.asarray(pairs)
        lines = shapely.linestrings(np.stack([points[pairs[:,0]],points[pairs[:,1]]],axis=1))
        # 끝점이안전해도연결선분이건물을가로지르면안됩니다. 대각선코너통과도여기서제거합니다.
        pairs = pairs[~shapely.intersects(lines,self.flight_blocker)]
        costs = np.linalg.norm(points[pairs[:,0]]-points[pairs[:,1]],axis=1)
        rows = np.r_[pairs[:,0],pairs[:,1]]; cols = np.r_[pairs[:,1],pairs[:,0]]
        graph = csr_matrix((np.r_[costs,costs],(rows,cols)),shape=(len(points),len(points)))
        # 서로끊어진비행공간에서가장큰성분만사용합니다. 제외비율을숨기지않고결과에남깁니다.
        n_components,labels = connected_components(graph,directed=False)
        sizes = np.bincount(labels); keep = np.flatnonzero(labels==np.argmax(sizes))
        self.points = points[keep]; self.graph = graph[keep][:,keep]
        self.tree = cKDTree(self.points)
        self.flight_free_fraction = float(free.mean())
        self.retained_free_cell_fraction = len(keep)/len(points)
        self.free_grid_components = int(n_components)
        self.distance,self.predecessor = shortest_path(self.graph,directed=False,return_predecessors=True)
        self.distance = self.distance.astype(np.float32)
        self.route = self.make_sweep(scenario)
        self.sensor_radius_m = scenario.sensor_radius_m
        self.sensor_occlusion = scenario.sensor_occlusion
        self.sensor_cache = {}

    # 원하는 좌표에서 가장 가까운 비행 가능 격자 노드를 찾습니다. 건물 안의 목표를 그대로 비행하지 않습니다.
    def nearest(self, point):
        return int(self.tree.query(point)[1])

    # 미리 계산한 predecessor를 역추적하여 안전한 최단 경로의 노드 목록을 복원합니다.
    def path(self, start, end):
        start,end = int(start),int(end)
        if start == end:
            return [start]
        nodes = [end]
        while nodes[-1] != start:
            pred = int(self.predecessor[start,nodes[-1]])
            if pred < 0:
                raise ValueError("서로 다른 비행 성분의 경로를 요청했습니다.")
            nodes.append(pred)
        return nodes[::-1]

    # 현재 노드에서 목표를 향해 한 간선만 이동할 다음 노드를 얻습니다. 대각선도 안전 검사된 간선만 사용합니다.
    def next_node(self, current, goal):
        return int(current) if current==goal else int(self.predecessor[goal,current])

    # 대형·대수·중계수와 독립적인 지그재그 왕복 기준 경로입니다.
    # 줄 간격은 기본40m이며 건물 우회 때문에 실제 경로 길이는 지역별로 달라집니다.
    def make_sweep(self, scenario):
        # 대형·드론 수·중계 수와 독립적인 왕복 수색 기준 경로를 만든다.
        desired = []
        for row,y in enumerate(np.arange(-160,161,scenario.sweep_spacing_m)):
            xs = (-160,160) if row%2==0 else (160,-160)
            for x in xs:
                desired.append((x,y) if scenario.direction=="east_west" else (y,x))
        goals = [self.nearest(q) for q in desired]
        route = [goals[0]]
        for goal in goals[1:]:
            route.extend(self.path(route[-1],goal)[1:])
        return np.array(self.points[route],float)

    # 반경25m 안의 실외 지면 셀에서 건물 시선 차폐를 통과한 셀 ID만 관측합니다.
    # 사람 검출·영상 인식·실내 수색 성공률을 뜻하지 않습니다.
    def sensing(self, position, scenario):
        # 센서는 실외 지면 셀을 관측한다. 영상 인식 성공률이나 실내 수색은 모사하지 않는다.
        key = (round(float(position[0]),5),round(float(position[1]),5),scenario.altitude_m)
        if key in self.sensor_cache:
            return self.sensor_cache[key]
        nearby = np.flatnonzero(np.linalg.norm(self.target_points-position[:2],axis=1)<=self.sensor_radius_m)
        if len(nearby) and self.sensor_occlusion:
            ground = np.column_stack([self.target_points[nearby],np.zeros(len(nearby))])
            _,los,_ = self.radio.propagation(position[None,:],ground,scenario)
            nearby = nearby[los]
        self.sensor_cache[key] = nearby
        return nearby

    # 지도 해시·건물 수·추정 높이 수·목표 셀 수·비행 성분 통계를 결과에 기록합니다.
    def summary(self):
        return {"region":self.key,"map_sha256":self.sha256,"buildings":len(self.buildings),
                "imputed_buildings":int(self.imputed.sum()),"target_cells":len(self.target_points),
                "flight_nodes":len(self.points),"free_grid_components":self.free_grid_components,
                "retained_free_cell_fraction":self.retained_free_cell_fraction,
                "flight_free_fraction":self.flight_free_fraction}
