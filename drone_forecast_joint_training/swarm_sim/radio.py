"""기하학적 차폐 + 공개 수식의 경량 전파·BPSK 패킷 모형.

ITU-R P.525 자유공간 손실, P.526-16 §4.5.1 Bullington 부분을 사용한다.
건물 지붕 프로파일에 적용한 단거리 평지 근사이며 레이트레이싱이나
3GPP TR 36.777 전체 구현이 아니다. BER 모형도 실제 무선 칩의 실측 곡선이 아니다.
"""
# 채널 모델의 원래 수식 구현입니다.
# 거리 → 자유공간 손실 + 건물 지붕 회절 → 수신 전력 → SINR → 패킷 성공 확률.
# LoS는 두 점 사이 직접 시선이 가려지지 않은 상태, NLoS는 가려진 상태입니다.
# 경로 손실은 ITU-R P.525와 P.526-16 §4.5.1의 평지 Bullington 부분을 참고합니다.
# 3GPP TR 36.777·ns-3·실제 Wi-Fi 칩 성능곡선 구현으로 해석하면 안 됩니다.
# 행 i는 송신 드론, 열 j는 수신 드론이며 전력 합산은 반드시 mW에서 합니다.
import numpy as np
import shapely
from shapely.strtree import STRtree
from scipy.special import erfc

C = 299792458.0

# 로그 단위 dBm을 선형 전력 mW로 바꿉니다. 간섭 전력과 잡음은 이 단위에서 더합니다.
def dbm_to_mw(value):
    return 10.0 ** (np.asarray(value, dtype=float) / 10.0)

# 선형 전력 mW를 dBm으로 되돌립니다. log(0)을 피하도록 매우 작은 양수 하한을 둡니다.
def mw_to_dbm(value):
    return 10.0 * np.log10(np.maximum(value, 1e-30))

# ITU P.525 자유공간 손실 L=20log10(4πdf/c)입니다. d는3차원 거리(m), f는Hz입니다.
# 0거리 발산을 피하려고 최소1m를 적용합니다. 거리만 반영하므로 건물 회절은 아래에서 추가합니다.
def fspl(distance_m, frequency_hz):
    return 20.0 * np.log10(4.0 * np.pi * np.maximum(distance_m, 1.0) * frequency_hz / C)

# P.526의 칼날 회절 근사 J(ν)입니다. ν는 지붕이 시선/프레넬 영역을 얼마나 침범하는지 나타냅니다.
# ν≤−0.78이면 이 근사에서 추가 손실을0dB로 둡니다.
def knife_loss(nu):
    nu = np.asarray(nu, dtype=float)
    good = nu > -0.78
    result = np.zeros_like(nu)
    x = nu[good] - 0.1
    result[good] = 6.9 + 20.0 * np.log10(np.sqrt(x*x + 1.0) + x)
    return result

# 부호화 없는 coherent BPSK에서 BER=erfc(sqrt(SINR×B/Rb))/2를 계산합니다.
# 비트 오류 독립을 가정해 성공 확률을(1−BER)^패킷비트수로 변환합니다. 실제 무선 장비 측정 곡선은 아닙니다.
def packet_success(sinr_db, n_bits, bandwidth_hz, bit_rate_bps):
    # 비트 에너지/잡음 밀도 = SINR × B/Rb. 부호화 없는 coherent BPSK 가정.
    ebn0 = 10.0 ** (np.clip(sinr_db, -100, 100)/10.0) * bandwidth_hz / bit_rate_bps
    ber = 0.5 * erfc(np.sqrt(ebn0))
    return np.exp(n_bits * np.log1p(-np.minimum(ber, 0.5)))

# 건물 원래 폴리곤과 높이로 송수신 경로의 차폐·회절을 계산합니다. 비행 안전 버퍼는 쓰지 않습니다.
class BuildingRadio:
    # 공간 인덱스 STRtree를 만들어 매 링크마다 모든 건물을 검사하는 비용을 줄입니다.
    def __init__(self, geometries, heights):
        self.geometries = np.asarray(geometries, dtype=object)
        self.heights = np.asarray(heights, dtype=float)
        self.tree = STRtree(self.geometries)
        self._propagation_cache = None
        self._propagation_cache_limit = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._a2a_tree = None

    def enable_cache(self, max_entries=200000):
        """같은 정적 지도에서 정확히 같은 좌표·전파 설정만 재사용.

        좌표 반올림·격자 스냅·송수신 방향 교환을 하지 않는다. 단일 도시
        객체 수명 동안의 유한 캐시이며 다른 지도 사이에서 공유하지 않는다.
        """
        if max_entries < 1:
            raise ValueError('전파 캐시 크기는 양수여야 합니다.')
        self._propagation_cache = {}
        self._propagation_cache_limit = int(max_entries)
        self.cache_hits = self.cache_misses = 0
        keep = self.heights >= 40.0
        self._a2a_geometries = self.geometries[keep]
        self._a2a_heights = self.heights[keep]
        self._a2a_tree = STRtree(self._a2a_geometries)

    # 송수신 좌표 N×3를 받아 손실dB·LoS여부·추가손실dB를 반환합니다.
    # 건물 교차 구간의 양 끝과 중간에서 시선 높이를 비교하고 Bullington 대표 회절점을 만듭니다.
    # 평지·지붕 프로파일 근사이며 재질별 투과·반사·지구곡률은 계산하지 않습니다.
    def propagation(self, tx, rx, scenario):
        if self._propagation_cache is None:
            return self._propagation_uncached(tx, rx, scenario)
        tx, rx = np.broadcast_arrays(np.asarray(tx, float), np.asarray(rx, float))
        tx = tx.reshape(-1, 3); rx = rx.reshape(-1, 3)
        pairs = np.ascontiguousarray(np.concatenate((tx, rx), axis=1), dtype=np.float64)
        if not len(pairs):
            return self._propagation_uncached(tx, rx, scenario)
        raw = pairs.view(np.dtype((np.void, 48))).reshape(-1)
        unique, first, inverse = np.unique(raw, return_index=True, return_inverse=True)
        config = (float(scenario.frequency_hz), scenario.radio_model, float(scenario.nlos_penalty_db))
        keys = [(config, item.tobytes()) for item in unique]
        cached = [self._propagation_cache.get(key) for key in keys]
        missing = [i for i, value in enumerate(cached) if value is None]
        self.cache_hits += len(pairs) - len(missing)
        self.cache_misses += len(missing)
        if missing:
            ids = first[missing]
            loss, los, extra = self._propagation_uncached(tx[ids], rx[ids], scenario)
            if len(self._propagation_cache) + len(missing) > self._propagation_cache_limit:
                self._propagation_cache.clear()
            for i, a, b, c in zip(missing, loss, los, extra):
                value = (float(a), bool(b), float(c))
                cached[i] = value
                if len(self._propagation_cache) < self._propagation_cache_limit:
                    self._propagation_cache[keys[i]] = value
        values = np.asarray(cached)
        return values[inverse, 0], values[inverse, 1].astype(bool), values[inverse, 2]

    def _propagation_uncached(self, tx, rx, scenario):
        tx, rx = np.broadcast_arrays(np.asarray(tx, float), np.asarray(rx, float))
        tx = tx.reshape(-1, 3); rx = rx.reshape(-1, 3)
        horizontal = np.linalg.norm(rx[:, :2] - tx[:, :2], axis=1)
        distance = np.linalg.norm(rx - tx, axis=1)
        # 고도 50m·2.4GHz·수평 거리 566m 이하에서 40m 미만 지붕은
        # knife_loss의 0 구간에 들어간다. 지상 간섭원 경로에는 적용 금지.
        # 끝점 높이와 거리를 매번 검사하며 좌표나 전파 수식은 근사하지 않는다.
        a2a = (self._a2a_tree is not None and scenario.frequency_hz == 2.4e9
               and np.all(tx[:, 2] == 50.0) and np.all(rx[:, 2] == 50.0)
               and np.all(horizontal <= 566.0))
        tree = self._a2a_tree if a2a else self.tree
        geometries = self._a2a_geometries if a2a else self.geometries
        heights = self._a2a_heights if a2a else self.heights
        count = len(tx)
        los = np.ones(count, dtype=bool)
        stim = np.full(count, -np.inf); srim = np.full(count, -np.inf)
        vmax = np.full(count, -np.inf)
        # 수평선분과겹치는건물만찾은다음, 그교차구간에서지붕과3차원시선의높이를비교합니다.
        lines = shapely.linestrings(np.stack([tx[:, :2], rx[:, :2]], axis=1))
        query = tree.query(lines, predicate="intersects")
        if query.shape[1]:
            li, bi = query
            intersections = shapely.intersection(lines[li], geometries[bi])
            pieces, parent = shapely.get_parts(intersections, return_index=True)
            keep = (shapely.get_type_id(pieces) == 1) & ~shapely.is_empty(pieces)
            pieces, parent = pieces[keep], parent[keep]
            li, bi = li[parent], bi[parent]
            a = shapely.line_locate_point(lines[li], shapely.get_point(pieces, 0))
            b = shapely.line_locate_point(lines[li], shapely.get_point(pieces, -1))
            low, high = np.minimum(a, b), np.maximum(a, b)
            mid = np.clip(horizontal[li]/2.0, low, high)
            for s in (low, high, mid):
                total = np.maximum(horizontal[li], 1e-6)
                s = np.clip(s, 1e-4, np.maximum(total - 1e-4, 1e-4))
                roof = heights[bi]
                ray = tx[li, 2] + (rx[li, 2] - tx[li, 2]) * s / total
                blocked = roof >= ray - 1e-9
                los[li[blocked]] = False
                np.maximum.at(stim, li, (roof - tx[li, 2]) / s)
                np.maximum.at(srim, li, (roof - rx[li, 2]) / np.maximum(total - s, 1e-4))
                nu = (roof-ray) * np.sqrt(2*total/((C/scenario.frequency_hz)*s*np.maximum(total-s, 1e-4)))
                np.maximum.at(vmax, li, nu)
        # 송수신방향의최대지붕기울기로Bullington대표회절점을정합니다. P.526전체complete method는아닙니다.
        slope = (rx[:, 2] - tx[:, 2]) / np.maximum(horizontal, 1e-6)
        blocked = stim >= slope
        nu = vmax.copy()
        regular = blocked & np.isfinite(stim) & (np.abs(stim + srim) > 1e-12)
        if np.any(regular):
            ids = np.flatnonzero(regular); total = np.maximum(horizontal[ids], 1e-6)
            db = (rx[ids, 2] - tx[ids, 2] + srim[ids]*total)/(stim[ids]+srim[ids])
            db = np.clip(db, 1e-4, np.maximum(total - 1e-4, 1e-4))
            above = tx[ids, 2] + stim[ids]*db - (tx[ids, 2]*(total-db)+rx[ids, 2]*db)/total
            nu[ids] = above * np.sqrt(2*total/((C/scenario.frequency_hz)*db*np.maximum(total-db, 1e-4)))
        j = knife_loss(nu)
        # 표준식의거리항은km입니다. 코드horizontal은m이므로1000으로나눕니다.
        bullington = j + (1-np.exp(-j/6.0))*(10+0.02*horizontal/1000.0)
        if scenario.radio_model == "fspl":
            extra = np.zeros(count)
        elif scenario.radio_model == "nlos_penalty":
            extra = np.where(los, 0.0, scenario.nlos_penalty_db)
        else:
            extra = bullington
        return fspl(distance, scenario.frequency_hz)+extra, los, extra

    # 현재 드론 위치에서 모든 방향 링크의 수신 전력·SINR·데이터/ACK 성공과 가용 그래프를 계산합니다.
    # 주로 초기 엔진과 검사에 쓰며, 전체 엔진은 같은 전파 수식과 fast_network의 고정 패킷 구현을 연결합니다.
    def state(self, positions, scenario, shadow_db, jammer_xyz=None, jammer_on=False):
        n = len(positions)
        tx = np.repeat(positions, n, axis=0); rx = np.tile(positions, (n, 1))
        loss, los, extra = self.propagation(tx, rx, scenario)
        loss = loss.reshape(n, n); los = los.reshape(n, n); extra = extra.reshape(n, n)
        received_dbm = scenario.tx_power_dbm + 2*scenario.gain_dbi - loss - shadow_db
        received_mw = dbm_to_mw(received_dbm)
        noise_dbm = -174.0 + 10*np.log10(scenario.bandwidth_hz) + scenario.noise_figure_db
        interference = np.zeros(n)
        if jammer_on and jammer_xyz is not None and scenario.jammer_power_dbm is not None:
            jloss, _, _ = self.propagation(np.asarray(jammer_xyz)[None, :], positions, scenario)
            interference = dbm_to_mw(scenario.jammer_power_dbm + scenario.gain_dbi - jloss)
        # 잡음+외부간섭은mW합산합니다. dBm값을그냥더하면물리적으로틀린계산이됩니다.
        denominator = dbm_to_mw(noise_dbm) + interference
        # 열 j는 수신 드론 j. dBm을 직접 더하지 않고 mW에서 합산한다.
        sinr = received_dbm - mw_to_dbm(denominator)[None, :]
        data_bits = 8*(scenario.packet_payload_bytes + scenario.packet_header_bytes)
        ps_data = packet_success(sinr, data_bits, scenario.bandwidth_hz, scenario.bit_rate_bps)
        ps_ack = packet_success(sinr.T, 8*scenario.ack_bytes, scenario.bandwidth_hz, scenario.bit_rate_bps)
        ps_hop = ps_data * ps_ack
        np.fill_diagonal(ps_hop, 0.0)
        adj = (ps_hop >= scenario.link_success_threshold) & (ps_hop.T >= scenario.link_success_threshold)
        np.fill_diagonal(adj, False)
        return {"loss_db":loss, "los":los, "extra_loss_db":extra, "rx_dbm":received_dbm,
                "rx_mw":received_mw, "noise_dbm":float(noise_dbm), "denominator_mw":denominator,
                "interference_mw":interference, "sinr_db":sinr, "ps_hop":ps_hop, "adjacency":adj}
