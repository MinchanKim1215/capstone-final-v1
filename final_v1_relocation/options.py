"""실험 설정. 개선안마다 끄고 켤 수 있고, 모두 끄면 V21과 같다."""
from dataclasses import asdict, dataclass


EVIDENCE = ("none", "model", "persistent", "both")


@dataclass(frozen=True)
class Options:
    # 개선안 1. 단절 중 복구 재배치
    cut_recovery: bool = False
    lost_age_db: float = .25          # 소식이 끊긴 1초마다 그 수색기 링크 여유를 이만큼 낮춘다 (V17 Forecast와 같은 계수)
    lost_horizon_db: float = .20      # 예측 시점이 1초 멀어질 때마다 추가로 낮춘다
    min_lost_age_s: float = 3.        # known 판정 기준(3초)보다 짧은 나이는 쓰지 않는다
    move_caps: tuple = ((6., 80.), (12., 57.))   # (끊긴 시간 이하, 허용 이동 폭 m) 순서
    move_cap_far_m: float = 40.       # 12초보다 오래 끊기면 이 폭까지만 바꾼다
    toward_moves: int = 4             # 끊긴 수색기 쪽 후보를 중계기마다 몇 개 더 볼지
    early_drop_db: float = .25        # 이미 연결된 수색기끼리의 0.5~2.5초 여유 하락 허용치
    min_deficit_gain: float = 1.5     # 재연결 시각이 같을 때 필요한 부족분 감소(dB·s)
    conservative_lost_noise: bool = True
    # 불필요한 복구 이동 줄이기: 복구를 실행하기 전에 요구하는 단절 근거
    #  none       : 근거 없이 실행 (처음 개선안 1)
    #  model      : 불확실성 보정 없이 계산한 모델이 지금(t=0) 끊겼다고 볼 때만
    #  persistent : 같은 수색기 소식이 직전 판단 주기(3초 전)에도 끊겨 있었을 때만
    #  both       : 두 조건을 모두 만족할 때만
    cut_evidence: str = "none"

    # 개선안 3. 발동 기준 조정
    threshold_scale: float = 1.       # V21 문턱 × 배율. 1이면 V21과 같다
    ambiguous_small_moves: bool = False
    small_move_m: float = 28.3        # 애매한 구간에서 허용할 보정 변화 (대각 20m 한 칸)

    # 개선안 4. 급락형 대비 우회 경로 확보
    bypass: bool = False
    shield_db: float = 6.             # 끊기면 망이 갈라지는 링크의 건물 감쇠가 이 이상이면 발동
    bypass_target_db: float = 0.      # 링크 하나를 지워도 남는 여유 목표
    bypass_min_gain_db: float = 3.
    bypass_move_cap_m: float = 40.
    bypass_distance_weight: float = .02
    bypass_cooldown_s: float = 6.

    def __post_init__(self):
        if self.cut_evidence not in EVIDENCE:
            raise ValueError(f"cut_evidence must be one of {EVIDENCE}")

    def move_cap_m(self, age_s):
        for limit, cap in self.move_caps:
            if age_s <= limit:
                return float(cap)
        return float(self.move_cap_far_m)

    def as_dict(self):
        return asdict(self)


# 확인 비행 42개에서 비교할 설정. 개선안 3은 문턱을 4단계로 낮춘다.
ARMS = {
    "v21": Options(),
    "r1_cut_recovery": Options(cut_recovery=True),
    "r1_model": Options(cut_recovery=True, cut_evidence="model"),
    "r1_persistent": Options(cut_recovery=True, cut_evidence="persistent"),
    "r1_both": Options(cut_recovery=True, cut_evidence="both"),
    "r3_t090": Options(threshold_scale=.9),
    "r3_t080": Options(threshold_scale=.8),
    "r3_t070": Options(threshold_scale=.7),
    "r3_t060": Options(threshold_scale=.6),
    "r3_t070_small": Options(threshold_scale=.7, ambiguous_small_moves=True),
    "r4_bypass": Options(bypass=True),
    "r13_combined": Options(cut_recovery=True, threshold_scale=.8, ambiguous_small_moves=True),
}

# 개선안 2. 명령을 못 받은 중계기의 대응
#  neutral   : 기본 대형 복귀 (Final V1 채택)
#  hold      : 직전 명령 유지 (최대 reconnect_window_s)
#  reconnect : 마지막으로 명령을 받은 위치 쪽으로 복귀, 시간 안에 다시 못 받으면 기본 대형
#  local     : Final V1의 자율 보정 (직전 명령 1주기 유지 + 로컬 무선 후보)
FALLBACKS = ("neutral", "hold", "reconnect", "local")
RECONNECT_WINDOW_S = 6.

# 복구 근거 선택용 개발 비행: 확인 비행과 겹치지 않는 학습 지도 5곳 × 간섭 3 × 난수 3
DEV_MAPS = ("bundang", "gangnam", "hongdae", "sangam_dmc", "teheran")
DEV_SEEDS = 3
DEV_ARMS = ("v21", "r1_cut_recovery", "r1_model", "r1_persistent", "r1_both")

# 명령 손실 14조건에서 비교할 (재배치 설정, 미수신 대응)
STRESS_ARMS = (
    ("v21", "neutral"),
    ("v21", "hold"),
    ("v21", "reconnect"),
    ("r1_cut_recovery", "neutral"),
    ("r1_cut_recovery", "reconnect"),
)
