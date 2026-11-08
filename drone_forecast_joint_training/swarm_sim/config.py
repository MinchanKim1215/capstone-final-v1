"""설계 변수, 환경 변수, 비교 조건을 구분하고 실행 전 유효성을 확인한다."""
# 변수의 단위를 이름 끝에 표시합니다. m=미터, s=초, Hz=헤르츠, bps=bit/s.
# Design은 비교할 배치, Scenario는 공간·센서·채널·패킷 조건입니다.
# 이 클래스는 초기 엔진과 공유하므로 필드가 있다고 전체 임무 엔진에서 모두 바꿀 수 있는 것은 아닙니다.
# 현재 전체 실험은 full_study.py의 7200초 상한·3초 이동과 full_run.py의 고정 설정 검사를 따릅니다.
from dataclasses import dataclass, asdict, fields
import math

FORMATIONS = ("line_cross", "line_along", "v", "grid", "ring")

# 직접 비교하는 설계 변수입니다. frozen=True로 실행 도중 설정이 몰래 바뀌지 않게 합니다.
@dataclass(frozen=True)
class Design:
    n_drones: int = 6
    n_relays: int = 2
    formation: str = "grid"
    spacing_m: float = 40.0
    formation_scope: str = "search"
    relay_layout: str = "backbone"

    # 수색 수 Ns=N−Nr입니다. 중계 계수는 Nr/N이며 Nr/Ns와 다릅니다.
    @property
    def n_search(self):
        return self.n_drones - self.n_relays

    # 총6~10대와 수색 최소2대 등 실행 가능한 범위를 확인합니다.
    # 후보 선정의 최소 수색3대는 실행 가능 조건과 별개이며 비교 화면에서 추가 적용합니다.
    def validate(self):
        if not isinstance(self.n_drones,int) or not isinstance(self.n_relays,int):
            raise ValueError("드론 수와 중계 수는 정수여야 합니다.")
        if not (6 <= self.n_drones <= 10 and 0 <= self.n_relays <= self.n_drones - 2):
            raise ValueError("총 6~10대, 중계 0~N-2대, 수색 최소 2대 조건을 지켜야 합니다.")
        if self.formation not in FORMATIONS or self.formation_scope not in ("search", "fleet"):
            raise ValueError("지원하지 않는 대형 또는 대형 적용 범위입니다.")
        if self.relay_layout not in ("backbone", "center"):
            raise ValueError("중계 기준 배치는 backbone 또는 center입니다.")
        if not math.isfinite(self.spacing_m) or self.spacing_m < 10:
            raise ValueError("기준 간격은 10m 이상 유한한 값이어야 합니다.")
        return self

# 공통 비교 조건을 담습니다. 기본120초·0.5초 이동은 초기 엔진용입니다.
# 전체 임무는 full_study.py에서7200초 상한·3초 이동으로 지정하고 mission.py에서 완료 시 종료합니다.
@dataclass(frozen=True)
class Scenario:
    # 초기 엔진 기본값입니다. 전체 실험은7200초 상한·3초 이동으로 덮어씁니다.
    duration_s: float = 120.0
    motion_dt_s: float = 0.5
    grid_m: float = 10.0
    # 전체 실험은50m고도 고정입니다. 높이50m이상건물+수평5m버퍼를비행장애물로둡니다.
    altitude_m: float = 50.0
    horizontal_clearance_m: float = 5.0
    vertical_clearance_m: float = 0.0
    min_separation_m: float = 5.0
    speed_mps: float = 5.0
    reference_speed_mps: float = 2.5
    sweep_spacing_m: float = 40.0
    direction: str = "east_west"
    route_start_fraction: float = 0.0
    sensor_radius_m: float = 25.0
    sensor_occlusion: bool = True
    height_case: str = "delivered"
    radio_model: str = "bullington"
    # 현재2.4GHz·10dBm·1MHz·1Mbps·잡음지수7dB 고정이며 열잡음은−107dBm입니다.
    frequency_hz: float = 2.4e9
    tx_power_dbm: float = 10.0
    gain_dbi: float = 0.0
    bandwidth_hz: float = 1e6
    noise_figure_db: float = 7.0
    bit_rate_bps: float = 1e6
    nlos_penalty_db: float = 20.0
    shadow_sigma_db: float = 1.5
    shadow_correlation_s: float = 4.0
    # 링크별데이터와역방향ACK의홉성공확률기준입니다. 임무연결유지율80/95%와는다른문턱입니다.
    link_success_threshold: float = 0.9
    packet_payload_bytes: int = 512
    packet_header_bytes: int = 64
    ack_bytes: int = 14
    slot_s: float = 0.005
    mac: str = "tdma"
    aloha_attempt_probability: float = 0.15
    routing: str = "expected_attempts"
    # 이1초필드는초기엔진용입니다. 전체엔진의실제라우팅주기는radio_dt_s(3초/0.5초)입니다.
    route_update_s: float = 1.0
    total_packets_per_s: float = 30.0
    traffic_mode: str = "fixed_total"
    per_search_packets_per_s: float = 5.0
    queue_limit: int = 64
    max_hop_attempts: int = 4
    packet_lifetime_s: float = 2.0
    timely_deadline_s: float = 0.25
    # 현재외부간섭원은0dBm·높이2m·임무30~70%구간연속송신입니다. 실측도시간섭지도가아닙니다.
    jammer_power_dbm: float | None = 0.0
    jammer_height_m: float = 2.0
    jammer_start_fraction: float = 0.30
    jammer_end_fraction: float = 0.70
    jammer_duty: float = 1.0
    relay_failure_fraction: float | None = None

    # 유한한 값·확률·시간 정합성과 한 슬롯의 데이터/ACK 직렬화 시간을 검사합니다.
    # 이 검사를 통과해도 전체 고속 엔진의 추가 고정 조건 검사를 통과해야 합니다.
    def validate(self):
        positive = ("duration_s", "motion_dt_s", "grid_m", "altitude_m", "min_separation_m",
                    "speed_mps", "reference_speed_mps", "sensor_radius_m", "frequency_hz",
                    "bandwidth_hz", "bit_rate_bps", "slot_s", "route_update_s", "packet_lifetime_s",
                    "timely_deadline_s", "shadow_correlation_s", "sweep_spacing_m",
                    "total_packets_per_s", "per_search_packets_per_s")
        for key in positive:
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key}는 유한한 양수여야 합니다.")
        if self.mac not in ("tdma", "aloha") or self.routing not in ("hops", "expected_attempts"):
            raise ValueError("지원하지 않는 MAC 또는 경로 기준입니다.")
        if self.radio_model not in ("fspl", "nlos_penalty", "bullington"):
            raise ValueError("지원하지 않는 채널 모델입니다.")
        if self.direction not in ("east_west", "north_south") or self.height_case not in ("delivered", "imputed_minus15", "imputed_plus15"):
            raise ValueError("지원하지 않는 진행 방향 또는 높이 가정입니다.")
        if self.traffic_mode not in ("fixed_total", "per_search"):
            raise ValueError("트래픽 모드는 fixed_total 또는 per_search입니다.")
        if not 0 < self.link_success_threshold < 1 or not 0 < self.aloha_attempt_probability <= 1:
            raise ValueError("확률 설정이 유효하지 않습니다.")
        if not 0 <= self.jammer_start_fraction < self.jammer_end_fraction <= 1 or not 0 <= self.jammer_duty <= 1:
            raise ValueError("간섭 활성 구간·점유율 설정이 유효하지 않습니다.")
        if self.horizontal_clearance_m < 0 or self.vertical_clearance_m < 0 or self.shadow_sigma_db < 0:
            raise ValueError("안전 여유와 표준편차는 0 이상이어야 합니다.")
        if self.queue_limit < 1 or self.max_hop_attempts < 1 or self.packet_payload_bytes < 8:
            raise ValueError("큐·재전송·패킷 크기 설정이 유효하지 않습니다.")
        if self.timely_deadline_s > self.packet_lifetime_s:
            raise ValueError("적시 전달 기준은 패킷 수명 이하여야 합니다.")
        if self.relay_failure_fraction is not None and not 0 < self.relay_failure_fraction < 1:
            raise ValueError("중계 이탈 시점은 실험 길이의 0~1 사이여야 합니다.")
        if not 0 <= self.route_start_fraction < 1:
            raise ValueError("기준 경로 시작 위치는 전체 길이의 0~1 미만 비율입니다.")
        for key in ("tx_power_dbm", "gain_dbi", "noise_figure_db", "nlos_penalty_db",
                    "horizontal_clearance_m", "vertical_clearance_m", "shadow_sigma_db"):
            if not math.isfinite(getattr(self,key)):
                raise ValueError(f"{key}는 유한해야 합니다.")
        if self.jammer_power_dbm is not None and not math.isfinite(self.jammer_power_dbm):
            raise ValueError("간섭원 전력은 유한한 dBm 또는 null이어야 합니다.")
        # 데이터512B+헤더64B+ACK14B의비트직렬화시간을5ms슬롯안에담는지확인합니다.
        wire_s = 8 * (self.packet_payload_bytes + self.packet_header_bytes + self.ack_bytes) / self.bit_rate_bps
        if wire_s > self.slot_s:
            raise ValueError("한 슬롯이 데이터·헤더·ACK의 직렬화 시간보다 짧습니다.")
        if not math.isclose(self.motion_dt_s/self.slot_s, round(self.motion_dt_s/self.slot_s), abs_tol=1e-8):
            raise ValueError("운동 시간 간격은 패킷 슬롯 길이의 정수 배여야 합니다.")
        if not math.isclose(self.duration_s/self.motion_dt_s, round(self.duration_s/self.motion_dt_s), abs_tol=1e-8):
            raise ValueError("실험 길이는 운동 시간 간격의 정수 배여야 합니다.")
        return self

# 설정 오타를 알 수 없는 필드 오류로 잡습니다. 잘못된 키를 조용히 무시하지 않습니다.
def from_dict(cls, data):
    valid = {field.name for field in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"알 수 없는 설정: {sorted(unknown)}")
    return cls(**data).validate()

# 배치와 공통 조건을 JSON으로 저장할 수 있는 사전 형태로 묶습니다.
def as_config(design, scenario):
    return {"design": asdict(design), "scenario": asdict(scenario)}
