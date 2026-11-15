"""재배치 개선 평가에 필요한 파일만 골라 업로드할 수 있는 크기로 압축합니다.

이 파일 하나를 지도·모델이 들어 있는 프로젝트 폴더(adaptive_joint_expanded, forecast_learning 폴더가
있는 곳)에 복사한 뒤 실행하세요. Python 표준 라이브러리만 씁니다.

  python collect_for_claude.py               # 19MB 조각으로 나눔
  python collect_for_claude.py --part-mb 50  # 조각 크기 변경
  python collect_for_claude.py --root D:\\capstone  # 폴더 직접 지정
  python collect_for_claude.py --maps D:\\지도폴더  # 지도를 따로 보관한 폴더도 찾기 (여러 번 지정 가능)

결과는 같은 폴더의 claude_upload\\ 안에 생깁니다. 그 안의 파일을 모두 올리면 됩니다.
조각 파일은 따로 열 수 없습니다. 그대로 올려 주세요.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path, PureWindowsPath

CONFIRMATION_MAPS = (
    "gyeonggi_20261025_06", "gyeonggi_20261030_bh3_03", "gyeonggi_20261030_c_22", "gyeonggi_extra_20261026_08",
    "gyeonggi_extra_20261026_10", "gyeonggi_extra_20261026_11", "gyeonggi_extra_20261027_02",
    "gyeonggi_fast_20261027_19", "incheon_20261030_b2_10", "incheon_20261030_b2_24", "incheon_20261030_b2_29",
    "incheon_20261030_b2_31", "incheon_20261030_b2_33", "seoul_20261026_batch2_extra_05",
)
# Final V1 소스의 V21 잠금 해시. freeze.json의 sources가 이 값과 같아야 Final V1에서 모델이 열린다.
FINAL_V1_V21_SOURCES = {
    "control.py": "83dc1701eef4a2f373a2ce5d9e4a28ff0a4b1541e394d116ffb5a8cc5b05c79c",
    "training.py": "01bf3e3e41aa3aede2b20ac43323105f80363b1fdc8614898eb8b0ea1cb06a27",
}
V7_RUN = "adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031"
V7_CHECKPOINT = V7_RUN+"/checkpoints/s0000064.pt"
PLAN = "additional_maps_20261030/expanded_plan.json"


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


SKIP = {".git", ".venv", "__pycache__", "artifacts", "node_modules", "claude_upload"}


def map_index(roots, depth=8):
    """지도 폴더 이름(또는 '<이름>.scene.local.json') → 파일 경로."""
    index = {}
    for root in roots:
        base = len(root.parts)
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP] if len(Path(folder).parts)-base < depth else []
            for name in files:
                if name == "scene.local.json":
                    index.setdefault(Path(folder).name, Path(folder)/name)
                elif name.endswith(".scene.local.json"):
                    index.setdefault(name[:-len(".scene.local.json")], Path(folder)/name)
    return index


def opens(path):
    try:
        return path.is_file()
    except OSError:
        return False


def find_root(start):
    candidates = [start, *start.parents]
    for depth in ("*", "*/*", "*/*/*"):
        candidates += [p.parent for p in start.glob(depth+"/adaptive_joint_expanded") if p.is_dir()]
    for folder in candidates:
        if (folder/"adaptive_joint_expanded").is_dir() and (folder/"forecast_learning").is_dir():
            return folder
    return None


class Collector:
    def __init__(self, root, map_roots=()):
        self.root = root
        self.map_roots = [Path(p).resolve() for p in map_roots]
        self.files = {}
        self.missing = []
        self.warnings = []

    def add(self, relative, path=None, expected=None, required=True):
        path = Path(path) if path is not None else self.root/relative
        if not path.is_file():
            (self.missing if required else self.warnings).append(f"없음: {relative}")
            return False
        if expected is not None and sha(path) != expected:
            self.warnings.append(f"버전 다름 (해시 불일치): {relative}")
        self.files[relative] = path
        return True

    def v7(self):
        self.add("adaptive_joint_expanded/artifacts/protocol.json")
        expected = None
        best = self.root/V7_RUN/"best.json"
        if best.is_file():
            self.add(V7_RUN+"/best.json")
            expected = read(best).get("sha256")
        self.add(V7_CHECKPOINT, expected=expected)

    def v21(self):
        folder = "forecast_learning/artifacts"
        if not self.add(folder+"/freeze.json"):
            return
        freeze = read(self.root/folder/"freeze.json")
        self.add(folder+"/protocol.json", expected=freeze.get("protocol_sha256"))
        for name, expected in freeze.get("manifests", {}).items():
            self.add(f"{folder}/{name}_manifest.json", expected=expected)
        for meta in freeze.get("models", {}).values():
            name = PureWindowsPath(meta["path"]).name
            stored = Path(meta["path"])
            local = stored if stored.is_file() else self.root/folder/"models"/name
            self.add(f"{folder}/models/{name}", local, expected=meta.get("sha256"))
        for name, expected in freeze.get("sources", {}).items():
            if FINAL_V1_V21_SOURCES.get(name) != expected:
                self.warnings.append(f"이 V21 모델은 Final V1과 다른 코드({name})로 잠겨 있습니다. 더 최신 폴더가 있는지 확인하세요.")

    def maps(self):
        """지도 path가 지도를 만든 PC의 절대 경로여도 같은 지도를 이 폴더에서 찾는다."""
        if not self.add(PLAN):
            return
        entries = {m["key"]: m for m in read(self.root/PLAN)["maps"]}
        legacy = self.root/"drone_forecast_joint_training"
        index = None
        for key in CONFIRMATION_MAPS:
            if key not in entries:
                self.missing.append(f"지도 목록에 없음: {key}")
                continue
            entry = entries[key]
            raw = entry["path"]
            parts = PureWindowsPath(raw).parts
            found = legacy/raw if opens(legacy/raw) else None
            relative = "drone_forecast_joint_training/"+raw if found and not PureWindowsPath(raw).is_absolute() \
                else f"drone_forecast_joint_training/maps/additional/{key}/scene.local.json"
            if found is None and "additional_maps_20261030" in parts:
                tail = self.root.joinpath(*parts[parts.index("additional_maps_20261030"):])
                found = tail if opens(tail) else None
            if found is None:
                for option in (legacy/"maps"/"additional"/key/"scene.local.json", legacy/"maps"/f"{key}.json"):
                    if opens(option):
                        found = option
                        break
            if found is None:
                if index is None:
                    index = map_index([*self.map_roots, self.root, self.root.parent])
                found = index.get(key)
            if found is None:
                self.missing.append(f"지도 파일을 찾지 못함: {key} (scene.local.json)")
                continue
            self.add(relative, found, expected=entry.get("sha256"))

    def optional(self):
        for path in sorted((self.root/"final_v1/artifacts").glob("protocol.json")):
            self.add("final_v1/artifacts/protocol.json", path, required=False)
        for path in sorted((self.root/"final_v1/artifacts/confirm/v21").glob("*.json")):
            self.add(path.relative_to(self.root).as_posix(), path, required=False)


def write(collector, out, part_bytes):
    out.mkdir(exist_ok=True)
    for old in out.glob("claude_data*"):
        old.unlink()
    archive = out/"claude_data.zip"
    manifest = {relative: dict(bytes=path.stat().st_size, sha256=sha(path))
                for relative, path in sorted(collector.files.items())}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for relative, path in sorted(collector.files.items()):
            zf.write(path, relative)
        zf.writestr("claude_manifest.json", json.dumps(dict(files=manifest, warnings=collector.warnings),
                                                       ensure_ascii=False, indent=2))
    size = archive.stat().st_size
    if size <= part_bytes:
        return [archive]
    count = -(-size//part_bytes)
    parts = []
    with open(archive, "rb") as stream:
        for index in range(1, count+1):
            part = out/f"claude_data_part{index}of{count}.zip"
            part.write_bytes(stream.read(part_bytes))
            parts.append(part)
    archive.unlink()
    return parts


def main():
    parser = argparse.ArgumentParser(description="재배치 개선 평가용 파일 수집")
    parser.add_argument("--root", help="프로젝트 폴더 (생략하면 자동으로 찾음)")
    parser.add_argument("--part-mb", type=float, default=19.)
    parser.add_argument("--check", action="store_true", help="압축하지 않고 확인만")
    parser.add_argument("--maps", action="append", default=[], help="지도를 따로 보관한 폴더")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else find_root(Path.cwd().resolve())
    if root is None or not (root/"adaptive_joint_expanded").is_dir():
        sys.exit("프로젝트 폴더를 찾지 못했습니다. --root 로 adaptive_joint_expanded 폴더가 있는 경로를 지정하세요.")
    print("프로젝트 폴더:", root)
    for folder in args.maps:
        if not Path(folder).is_dir():
            sys.exit(f"지도 폴더가 없습니다: {folder}")
    collector = Collector(root, args.maps)
    collector.v7()
    collector.v21()
    collector.maps()
    collector.optional()
    total = sum(path.stat().st_size for path in collector.files.values())
    print(f"찾은 파일 {len(collector.files)}개, 압축 전 {total/2**20:.1f}MB")
    for relative, path in sorted(collector.files.items(), key=lambda kv: -kv[1].stat().st_size)[:8]:
        print(f"  {path.stat().st_size/2**20:8.1f}MB  {relative}")
    for line in collector.warnings:
        print("주의:", line)
    if collector.missing:
        print("\n필수 파일이 빠졌습니다. 아래 결과를 그대로 Claude에게 보여 주세요.")
        for line in collector.missing:
            print("  -", line)
    if args.check:
        return
    if collector.missing:
        answer = input("빠진 파일이 있어도 찾은 것만 압축할까요? [y/N] ").strip().lower()
        if answer != "y":
            return
    parts = write(collector, root/"claude_upload", int(args.part_mb*2**20))
    print(f"\n완료: {root/'claude_upload'} 안의 파일 {len(parts)}개를 모두 올려 주세요.")
    for part in parts:
        print(f"  {part.name}  {part.stat().st_size/2**20:.1f}MB")


if __name__ == "__main__":
    main()
