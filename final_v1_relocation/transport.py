"""개선안 2. 명령을 못 받은 중계기의 자율 재연결.

Final V1의 CommandChannel.apply()는 명령을 못 받은 중계기에 두 가지 대응만 둔다.
  neutral : 기본 대형 복귀 (Final V1 채택)
  local   : 직전 명령 1주기 유지 + 로컬 무선 후보 (기본 대형보다 불리했음)
여기에 두 가지를 더한다.
  hold      : 직전에 받은 명령을 최대 RECONNECT_WINDOW_S 동안 유지, 그 뒤 기본 대형
  reconnect : 마지막으로 명령을 받은 위치 쪽으로 돌아간다. RECONNECT_WINDOW_S 안에 다시 명령을
              받지 못하면 기본 대형으로 돌아간다. 수색기는 계속 움직이므로 과거 위치는 금방
              의미를 잃는다. 그래서 시간 제한을 둔다.

수신 여부는 부모 클래스가 같은 난수 순서로 뽑는다. 그래서 대응 방식끼리 같은 수신 표본
(공통 난수)에서 비교된다. 중계기는 자기 위치(GPS)와 공유 항법 계획의 대형 위치만 쓴다.
"""
import numpy as np

from adaptive_joint_expanded.features import OFFSETS
from final_v1.transport import CommandChannel

from .options import FALLBACKS, RECONNECT_WINDOW_S

STEP_S = 3.


class FallbackChannel(CommandChannel):
    def __init__(self, seed, n, ns, baseline_steps, mode, world, window_s=RECONNECT_WINDOW_S):
        if mode not in FALLBACKS:
            raise ValueError(f"Unknown command-loss fallback: {mode}")
        super().__init__(seed, n, ns, baseline_steps)
        self.mode = mode
        self.world = world
        self.window = int(round(window_s/STEP_S))
        self.anchor = np.full((n-ns, 2), np.nan)

    def apply(self, tick, probabilities, requested, observation, local_rf, fallback):
        if self.mode == "local":
            return super().apply(tick, probabilities, requested, observation, local_rf, "local")
        applied, received, hops, forced, reason = super().apply(
            tick, probabilities, requested, observation, local_rf, "neutral")
        if self.mode == "neutral":
            return applied, received, hops, forced, reason
        env = self.world()
        position = env.city.points[env.nodes[self.ns:]]
        self.anchor[received] = position[received]
        nominal = np.asarray(env.proposal["nominal"])[self.ns:]
        edge = 200.-env.scenario.grid_m/2
        for relay in np.flatnonzero(~received):
            gap = tick-self.last_tick[relay]
            if gap > self.window or not np.isfinite(self.anchor[relay]).all():
                reason[relay] = "nominal_timeout"
                continue
            valid = np.asarray(observation["action_mask"][relay], dtype=bool)
            if self.mode == "hold":
                old = int(self.last[relay])
                if valid[old]:
                    applied[relay], reason[relay] = old, "hold_last"
                continue
            targets = np.clip(nominal[relay]+OFFSETS, -edge, edge)
            distance = np.linalg.norm(targets-self.anchor[relay], axis=1)
            distance[~valid] = np.inf
            if np.isfinite(distance).any():
                applied[relay], reason[relay] = int(np.argmin(distance)), "return_anchor"
        return applied, received, hops, forced, reason
