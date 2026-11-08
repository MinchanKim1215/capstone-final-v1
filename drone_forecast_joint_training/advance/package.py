"""최종 코드·가중치·동일 결과·원시 시연을 묶고 다른 경로에서 검사."""

import argparse, json, shutil, subprocess, sys, zipfile
from pathlib import Path
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from advance.data import RESULT, DATA, BASE
from advance.train import RUNS

DEST = ROOT.parent.parent / "deliverables"
STAGE = DEST / "staging/drone_forecast_prediction_final"


def raw_archive():
    manifest = read_json(
        ROOT / "datasets/advance_confirmation/collection_manifest.json"
    )
    DEST.mkdir(exist_ok=True)
    path = DEST / "drone_prediction_fresh24_raw.zip"
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        f = ROOT / "datasets/advance_confirmation/collection_manifest.json"
        z.write(f, f.relative_to(ROOT).as_posix())
        for r in manifest["records"]:
            for name in [
                "observations.npz",
                "truth.npz",
                "spatial_observations.npz",
                "episode.json",
            ]:
                f = ROOT / r["path"] / name
                z.write(f, f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    out = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "flights": 24,
        "extract_at": "project root; paths start datasets/advance_confirmation",
    }
    write_json(RESULT / "fresh_archive.json", out)
    print(json.dumps(out))
    return path


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    def cp(file, relative=None):
        file = Path(file)
        r = Path(relative) if relative is not None else file.relative_to(ROOT)
        dest = STAGE / r
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)

    for folder in ["swarm_sim", "upgrade", "usable", "advance", "tests"]:
        for f in (ROOT / folder).glob("*"):
            if f.is_file() and (
                f.suffix in [".py", ".cjs"]
                or (f.suffix == ".html" and f.name.endswith("_template.html"))
            ):
                cp(f)
    for name in [
        "advance_run.py",
        "requirements-advance.txt",
        "00_환경설정.bat",
        "01_예측기_재생.bat",
        "DATA_INDEX.md",
    ]:
        cp(ROOT / name)
    cp(ROOT / "README_예측개선_최종.md", "README.md")
    for f in (ROOT / "configs").glob("advance*.json"):
        cp(f)
    cp(ROOT / "configs/usable_protocol.json")
    cp(ROOT / "configs/channel_core_reference.json")
    cp(ROOT / "results/usable/motion_check.json")
    protocol = read_json(ROOT / "configs/usable_protocol.json")
    for e in protocol["maps"]:
        cp(ROOT / e["path"])
    for f in (ROOT / "cache/usable").glob("*_motion.*"):
        cp(f)
    for folder in ["model/advance/selected", "model/advance/transformer_reference"]:
        for f in (ROOT / folder).glob("*"):
            if f.is_file():
                cp(f)
    for f in RESULT.glob("*.json"):
        if f.name not in ["work_status.json", "format_semantics.json", "fresh_raw_library_receipt.json"]:
            cp(f)
    for f in RESULT.glob("*_aggregate.npz"):
        cp(f)
    for name in ["selected", "transformer_reference"]:
        for f in (RESULT / name).glob("*.json"):
            cp(f)
    for folder in RUNS.glob("*_s*"):
        if not (folder / "run.json").exists():
            continue
        for f in folder.glob("*"):
            if (
                f.is_file()
                and f.name != "resume.pt"
                and f.suffix in [".pt", ".json", ".npy"]
            ):
                cp(f)
        for f in (folder / "heads").glob("*"):
            if f.is_file() and f.suffix in [".json", ".txt", ".npy"]:
                cp(f)
    for f in BASE.rglob("*"):
        if (
            f.is_file()
            and (
                f.parent == BASE
                or f.parent.name in ["dense", "runtime", "alarm_calibration_scores"]
            )
            and f.suffix in [".json", ".pt", ".npz"]
        ):
            cp(f)
    for f in DATA.rglob("*.json"):
        cp(f)
    for dataset in ["usable45", "enhanced_extra", "advance_confirmation"]:
        cp(ROOT / "datasets" / dataset / "collection_manifest.json")
    cases = {
        ("teheran", "east_west", "intermittent"),
        ("seoul_20261023_03", "north_south", "intermittent"),
        ("bucheon_jungdong_east", "east_west", "continuous"),
    }
    manifest = read_json(ROOT / "datasets/usable45/collection_manifest.json")
    for r in manifest["records"]:
        if (r["region"], r["metadata"]["direction"], r["metadata"]["kind"]) in cases:
            for name in [
                "observations.npz",
                "truth.npz",
                "spatial_observations.npz",
                "episode.json",
            ]:
                cp(ROOT / r["path"] / name)
    for name in ["통신장애_예측성능_최종검증.html", "통신위험_전체임무_최종.html"]:
        cp(ROOT / "docs" / name)
        shutil.copyfile(ROOT / "docs" / name, DEST / name)
    for f in (ROOT / "docs/assets").glob("advance_*.png"):
        cp(f)
    for name in ["NotoSansKR.ttf", "OFL-NotoSansKR.txt"]:
        cp(ROOT / "docs/assets" / name)
    for f in (RESULT / "replays").glob("*.csv"):
        cp(f)
    shutil.copyfile(ROOT / "DATA_INDEX.md", DEST / "예측실험_데이터_목록.md")
    with (DEST / "package_smoke.log").open("w") as log:
        for command in ["doctor", "verify"]:
            subprocess.run(
                [sys.executable, str(STAGE / "advance_run.py"), command],
                cwd=DEST,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        subprocess.run(
            [
                "node",
                str(STAGE / "advance/verify_replay.cjs"),
                str(STAGE / "docs/통신위험_전체임무_최종.html"),
            ],
            cwd=DEST,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    smoke = {
        "pass": True,
        "fresh_stage": True,
        "outside_working_directory": True,
        "doctor_and_real_predictor_and_tests": True,
        "replay_script": True,
        "raw_demo_flights": 3,
        "full_experiment_raw_data_in_basic_zip": False,
    }
    write_json(STAGE / "results/advance/package_smoke.json", smoke)
    write_json(RESULT / "package_smoke.json", smoke)
    entries = []
    for f in sorted(STAGE.rglob("*")):
        if (
            not f.is_file()
            or "__pycache__" in f.parts
            or f.name == "advance_last_runtime_check.json"
        ):
            continue
        entries.append(
            {
                "path": f.relative_to(STAGE).as_posix(),
                "bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            }
        )
    write_json(
        STAGE / "PACKAGE_CONTENTS.json",
        {
            "files": entries,
            "manifest_excludes_itself": True,
            "selected_checkpoint_sha256": read_json(RESULT / "selection.json")[
                "selected"
            ]["checkpoint_sha256"],
            "raw_demo_flights": 3,
            "all_study_flights": 474,
        },
    )
    return STAGE


def archive():
    path = DEST / "drone_forecast_prediction_final.zip"
    index = read_json(STAGE / "PACKAGE_CONTENTS.json")
    with zipfile.ZipFile(
        path, "w", zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True
    ) as z:
        for row in index["files"]:
            f = STAGE / row["path"]
            assert sha256_file(f) == row["sha256"]
            z.write(f, (Path(STAGE.name) / row["path"]).as_posix())
        z.write(STAGE / "PACKAGE_CONTENTS.json", STAGE.name + "/PACKAGE_CONTENTS.json")
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
    receipt = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "crc_verified": True,
        "files": len(index["files"]) + 1,
        "fresh_stage_smoke": True,
    }
    write_json(DEST / "package_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-only", action="store_true")
    p.add_argument("--archive-only", action="store_true")
    a = p.parse_args()
    if a.raw_only:
        raw_archive()
    else:
        if not a.archive_only:
            stage()
        archive()
