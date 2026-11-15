"""Final V1 비행 루프(final_v1.runner.run)를 그대로 쓰고 재배치 판단과 명령 채널만 교체한다."""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from final_v1 import runner as native

from .control import Controller
from .options import ARMS, FALLBACKS
from .paths import localize_maps
from .transport import FallbackChannel

CONTROLLERS = ("v21", "v21_transformer")


def localize_freeze():
    """freeze.json의 V21 모델 경로가 다른 PC의 절대 경로이면 이 폴더의 같은 파일을 쓴다.

    파일 내용의 SHA-256 검사(verify_freeze)는 그대로 거친다. 경로만 바꾼다.
    """
    import forecast_learning.control as learned
    from forecast_learning import ARTIFACTS as LEARNED

    original = learned.read
    if getattr(original, "localized", False):
        return

    def read(path):
        value = original(path)
        if Path(path).name == "freeze.json":
            for meta in value.get("models", {}).values():
                if not Path(meta["path"]).is_file():
                    local = LEARNED/"models"/PureWindowsPath(meta["path"]).name
                    if local.is_file():
                        meta["path"] = str(local)
        return value

    read.localized = True
    learned.read = read


def initialize(controller="v21"):
    """Final V1 fly.py와 같은 초기화. Transformer 업데이트는 자체 초기화를 쓴다."""
    localize_freeze()
    localize_maps()
    if controller == "v21":
        native.initialize()
    else:
        from fast_halfsec.context import initialize as upgraded

        upgraded()


class _World:
    """CommandChannel이 중계기 위치와 대형 위치를 읽을 수 있도록 현재 World를 기억한다."""

    def __init__(self):
        self.env = None

    def __call__(self):
        return self.env

    def subclass(self):
        holder = self

        class TrackedWorld(native.World):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                holder.env = self

        return TrackedWorld


@lru_cache(maxsize=1)
def _transformer():
    from final_v1.transformer_upgrade.release import installed

    payload, profile = installed()
    fleet = profile["thresholds"]["fleet"]
    return payload, fleet["threshold"], fleet["network_weight"]


def run(spec, arm="v21", transport=None, max_steps=None, controller="v21"):
    """arm: options.ARMS 키. transport: None 또는 options.FALLBACKS 중 하나."""
    if arm not in ARMS:
        raise ValueError(f"Unknown relocation arm: {arm}")
    if transport is not None and transport not in FALLBACKS:
        raise ValueError(f"Unknown command-loss fallback: {transport}")
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {controller}")
    decision = Controller(ARMS[arm])
    world = _World()
    with ExitStack() as stack:
        stack.enter_context(patch.object(native, "choose", decision.choose))
        if transport is not None:
            stack.enter_context(patch.object(native, "World", world.subclass()))
            stack.enter_context(patch.object(
                native, "CommandChannel",
                lambda seed, n, ns, steps: FallbackChannel(seed, n, ns, steps, transport, world)))
        if controller == "v21":
            row, data = native.run(spec, "v21", transport=transport, max_steps=max_steps)
        else:
            from final_v1.transformer_upgrade.bridge import run as upgraded

            payload, threshold, weight = _transformer()
            row, data = upgraded(spec, "v21_transformer", payload, threshold, transport=transport,
                                 max_steps=max_steps, network_weight=weight)
    row.update(relocation_arm=arm, relocation_controller=controller,
               relocation_options=ARMS[arm].as_dict(),
               v18_reasons=dict(decision.v18_reasons), relocation_counts=dict(decision.counts))
    return row, data
