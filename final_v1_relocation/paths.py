"""지도 경로 현지화.

expanded_plan.json의 일부 지도 path는 지도를 만든 PC의 절대 경로(C:\\Users\\...\\additional_maps_20261030\\raw\\...)로
저장돼 있다. 원본 코드는 LEGACY / path로 그 경로를 그대로 열기 때문에 다른 PC에서는 지도를 찾지 못한다.
여기서는 같은 지도를 이 작업 폴더에서 찾아 경로만 바꾼다. 지도 내용의 SHA-256 검사(entry_for)는 그대로 거친다.
"""
import os
from pathlib import Path, PureWindowsPath

from . import WORKSPACE

ANCHOR = "additional_maps_20261030"
NAME = "scene.local.json"


def opens(path, root=None):
    try:
        return ((root or WORKSPACE)/"drone_forecast_joint_training"/path).is_file()
    except OSError:
        return False


def candidates(entry, root):
    legacy = root/"drone_forecast_joint_training"
    raw = entry["path"]
    parts = PureWindowsPath(raw).parts
    key = entry["key"]
    yield legacy/raw
    if ANCHOR in parts:
        yield root.joinpath(*parts[parts.index(ANCHOR):])
    yield legacy/"maps"/"additional"/key/NAME
    yield legacy/"maps"/f"{key}.json"
    yield legacy/"maps"/f"{key}.{NAME}"


class MapIndex:
    """작업 폴더를 한 번 훑어 지도 이름 → 파일 경로를 만든다 (후보 경로에 없을 때만 쓴다)."""

    def __init__(self, roots):
        self.roots = roots
        self.index = None

    def find(self, key):
        if self.index is None:
            self.index = {}
            for root in self.roots:
                for folder, dirs, files in os.walk(root):
                    dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__", "artifacts", "node_modules")]
                    for name in files:
                        if name == NAME:
                            self.index.setdefault(Path(folder).name, Path(folder)/name)
                        elif name.endswith("."+NAME):
                            self.index.setdefault(name[:-len(NAME)-1], Path(folder)/name)
        return self.index.get(key)


def locate(entry, index=None, root=None):
    root = root or WORKSPACE
    for path in candidates(entry, root):
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return index.find(entry["key"]) if index is not None else None


def localize_maps():
    """adaptive_joint_expanded.contracts.plan()이 이 PC에서 찾은 지도 경로를 돌려주게 한다."""
    from adaptive_joint_expanded import contracts

    original = contracts.plan
    if getattr(original, "localized", False):
        return
    root = WORKSPACE
    index = MapIndex([root])
    found = {}

    def plan():
        value = original()
        for entry in value["maps"]:
            key = entry["key"]
            if key not in found:
                found[key] = None if opens(entry["path"], root) else locate(entry, index, root)
            if found[key] is not None:
                entry["path"] = str(found[key])
        return value

    plan.localized = True
    contracts.plan = plan


def localize_registry(registry):
    """forecast_rebuild.contracts.maps() 결과의 경로를 같은 방식으로 바꾼다 (단절 유형 집계용)."""
    index = MapIndex([WORKSPACE])
    for entry in registry.values():
        if not opens(entry["path"]):
            local = locate(entry, index)
            if local is not None:
                entry["path"] = str(local)
    return registry
