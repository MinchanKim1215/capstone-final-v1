"""중단된 NPZ·잘못된 체크포인트 결과를 재사용하지 않는 검증."""

from pathlib import Path
import json, zipfile
import numpy as np
from swarm_sim.regions import sha256_file


def validated_result(prefix, spec, checkpoint_sha=None, rollout=False):
    prefix = Path(prefix)
    meta = prefix.with_suffix(".json")
    arrays = prefix.with_suffix(".npz")
    if not meta.exists() or not arrays.exists():
        return None
    try:
        result = json.loads(meta.read_text(encoding="utf-8"))
        if any(result.get(k) != v for k, v in spec.items()):
            return None
        if result.get("checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
            return None
        if result.get("array_sha256", sha256_file(arrays)) != sha256_file(arrays):
            return None
        with zipfile.ZipFile(arrays) as archive:
            if archive.testzip() is not None:
                return None
        with np.load(arrays) as z:
            steps = int(round(result["duration_s"] / 3))
            if (
                z["nodes"].shape != (steps + 1, 8)
                or len(z["C"]) != steps * 6
                or len(z["actions"]) != steps
            ):
                return None
            if rollout:
                for name in (
                    "context",
                    "candidates",
                    "mask",
                    "teacher",
                    "state",
                    "chosen",
                    "logp",
                    "value",
                    "rewards",
                ):
                    if len(z[name]) != steps:
                        return None
                if (
                    not np.isfinite(z["logp"]).all()
                    or not np.isfinite(z["value"]).all()
                ):
                    return None
        return result
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        EOFError,
        json.JSONDecodeError,
    ):
        return None
