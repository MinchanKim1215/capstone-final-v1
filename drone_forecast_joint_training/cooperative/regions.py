"""기존 45개 분할을 보존하면서 같은 양식의 새 지도 추가."""

from pathlib import Path
import json, re, shutil
from swarm_sim.regions import ROOT, sha256_file
from swarm_sim.config import Scenario, Design
from usable.collect import prepare_city_without_unused_radio_cache
from usable.motion import full_mission
from swarm_sim.forecast import atomic_npz

EXTRA = ROOT / "configs/cooperative_extra_maps.json"


def entries():
    base = json.loads(
        (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
    )["maps"]
    extra = (
        json.loads(EXTRA.read_text(encoding="utf-8"))["maps"] if EXTRA.exists() else []
    )
    keys = [r["key"] for r in base + extra]
    if len(set(keys)) != len(keys):
        raise ValueError("지도 키 중복")
    return base + extra


def register(path):
    source = Path(path).resolve()
    scene = json.loads(source.read_text(encoding="utf8"))
    meta = scene["metadata"]
    key = meta["region_key"]
    if re.fullmatch("[a-z0-9_]+", key) is None:
        raise ValueError("region_key: 영문 소문자·숫자·밑줄 필요")
    existing = next((e for e in entries() if e["key"] == key), None)
    if existing:
        if existing["sha256"] == sha256_file(source):
            return dict(existing, reused=True)
        raise ValueError(
            "기존 지도 키의 내용 변경 금지. 새 region_key로 별도 등록 필요."
        )
    # 새 지도에서 높이 정보와 제공된 비행 제한이 서로 모순되면 등록하지 않음.
    # 기존 45개 지도에는 동일 검사를 별도 수행하여 모두 통과 확인.
    import shapely
    from shapely.geometry import shape, box
    import numpy as np

    if meta["local_bounds"] != [-200, -200, 200, 200]:
        raise ValueError("중심 원점의 400m×400m local_bounds 필요")
    heights = np.asarray([b["height_m"] for b in scene["buildings"]], float)
    if not np.isfinite(heights).all() or np.any(heights <= 0):
        raise ValueError("모든 건물의 유한한 양수 높이 필요")
    expected = shapely.union_all(
        [
            shape(b["geometry_local_m"])
            .buffer(5)
            .intersection(box(-200, -200, 200, 200))
            for b in scene["buildings"]
            if b["height_m"] >= 50
        ]
    )
    supplied = shapely.union_all(
        [shape(z["geometry_local_m"]) for z in scene["no_fly_zones"]]
    )
    if (
        shapely.area(shapely.difference(expected, shapely.buffer(supplied, 0.001)))
        > 0.01
    ):
        raise ValueError(
            "50m 이상 건물의 5m 안전 영역이 no_fly_zones에서 누락. 지도 비행 정책과 제한 폴리곤 확인 필요."
        )
    target = ROOT / "maps/custom" / key / "scene.local.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        shutil.copy2(source, target)
    # City가 좌표 범위·기하·높이·ID·비행 가능 성분을 검증.
    for direction in ("east_west", "north_south"):
        scenario = Scenario(
            duration_s=7200, motion_dt_s=3, direction=direction
        ).validate()
        city = prepare_city_without_unused_radio_cache(
            target, scenario, ROOT / "cache/usable"
        )
        mission = full_mission(city, Design(8, 5, "v", 40).validate(), scenario)
        if not mission["mission_completed"]:
            raise ValueError("기준 임무의 전체 수색 완료 실패: " + direction)
        file = ROOT / "cache/usable" / f"{key}_{direction}_motion.npz"
        atomic_npz(file, **{k: mission[k] for k in ("nodes", "desired", "phase")})
        file.with_suffix(".json").write_text(
            json.dumps(
                {
                    k: v
                    for k, v in mission.items()
                    if k not in ("nodes", "desired", "phase")
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    entry = {
        "key": key,
        "name": meta["region_name"],
        "path": target.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(target),
        "split": "custom_unseen",
        "source_path": source.name,
    }
    extra = (
        json.loads(EXTRA.read_text(encoding="utf-8"))
        if EXTRA.exists()
        else {"format": "cooperative-extra-maps-v1", "maps": []}
    )
    extra["maps"].append(entry)
    tmp = EXTRA.with_suffix(".tmp")
    tmp.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(EXTRA)
    return entry
