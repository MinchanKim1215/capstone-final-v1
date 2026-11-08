"""원시 파일·모델 동결·수치 표·전체 재생을 연결한 최종 검수."""

import json, subprocess, unittest
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from advance.data import RESULT
from advance.infer import selected
from advance.infer import eligible
from advance.features import read_record
from usable.data import index_episode


def main():
    protocol = read_json(ROOT / "configs/usable_protocol.json")
    records = []
    for dataset in ["usable45", "enhanced_extra", "advance_confirmation"]:
        m = read_json(ROOT / "datasets" / dataset / "collection_manifest.json")
        assert m["complete"]
        records += m["records"]
    assert len(records) == 474
    sets = {k: set(v) for k, v in protocol["splits"].items()}
    groups = {e["key"]: e["spatial_group"] for e in protocol["maps"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not sets[a] & sets[b] and not {groups[x] for x in sets[a]} & {
            groups[x] for x in sets[b]
        }
    maps = {e["key"]: e for e in protocol["maps"]}
    for e in maps.values():
        assert sha256_file(ROOT / e["path"]) == e["sha256"]
    assert len({r["metadata"]["seed"] for r in records}) == 474
    hash_rows = []
    sources = {}
    motions = set()
    minimum = 1000.0
    operational_count = 0
    operational_records = 0
    for q, r in enumerate(records):
        meta = r["metadata"]
        assert (
            meta["complete"]
            and not meta["recording_truncated"]
            and meta["mission"]["mission_completed"]
        )
        assert meta["map"]["map_sha256"] == maps[r["region"]]["sha256"]
        minimum = min(minimum, meta["mission"]["minimum_swept_separation_m"])
        motions.add((r["region"], meta["direction"]))
        for name, want in meta["files_sha256"].items():
            assert sha256_file(ROOT / r["path"] / name) == want, (r["id"], name)
        for name, want in meta["source_sha256"].items():
            if name not in sources:
                sources[name] = sha256_file(ROOT / name)
            assert sources[name] == want, (r["id"], name)
        # 보정·시험의 일괄 평가 선별이 실제 예측기의 지도/명령 지원 조건과 일치하는지 확인.
        if r["split"] != "train":
            index = index_episode(r, protocol)["index"]
            x, m, intent = read_record(r)
            use = eligible(x, m, index)
            k, i, j = index[use].T
            supported = m[k, i, j, 48:60].all(-1) & (intent[k, i, j, 8] > 0.5)
            assert supported.all(), (r["id"], "일괄 평가와 실제 관측 지원 조건 불일치", int((~supported).sum()))
            operational_count += int(use.sum())
            operational_records += 1
        hash_rows.append(
            {
                "id": r["id"],
                "region": r["region"],
                "split": r["split"],
                "files": meta["files_sha256"],
                "spatial_sha256": sha256_file(
                    ROOT / r["path"] / "spatial_observations.npz"
                ),
            }
        )
        if (q + 1) % 50 == 0:
            print("data audit", q + 1, len(records), flush=True)
    write_json(
        RESULT / "raw_integrity.json",
        {
            "records": hash_rows,
            "maps": 45,
            "completed_motion_paths": len(motions),
            "minimum_separation_m": minimum,
            "core_source_unchanged": True,
            "all_raw_hashes_verified": True,
            "spatial_groups_disjoint": True,
            "operational_support_gate_checked_records": operational_records,
            "operational_support_gate_checked_windows": operational_count,
            "batch_and_actual_predictor_support_conditions_match": True,
        },
    )
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful()
    path, _ = selected()
    modelsha = sha256_file(path)
    calsha = sha256_file(RESULT / "selected/calibration.json")
    policysha = sha256_file(RESULT / "selected/alarm_policy.json")
    assert sha256_file(ROOT / "model/advance/selected/best.pt") == modelsha
    for label in ["regression", "confirmation"]:
        f = read_json(RESULT / (label + "_frozen.json"))
        assert (
            f["checkpoint_sha256"] == modelsha
            and f["calibration_sha256"] == calsha
            and f["policy_sha256"] == policysha
        )
        r = read_json(RESULT / (label + "_evaluation.json"))
        assert not r["control_repositioning_implemented"]
    assert read_json(RESULT / "regression_evaluation.json")["samples"] == 4866832
    rt = read_json(RESULT / "runtime_verification.json")
    assert rt["checkpoint_sha256"] == modelsha
    bank = read_json(RESULT / "bank_parity.json")
    assert bank["maximum_feature_difference"] < 2e-5
    result = subprocess.run(
        [
            "node",
            str(ROOT / "advance/verify_replay.cjs"),
            str(ROOT / "docs/통신위험_전체임무_최종.html"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    replay = json.loads(result.stdout)
    write_json(RESULT / "replay_ui_verification.json", replay)
    out = {
        "pass": True,
        "tests_passed": tests.testsRun,
        "data_summary": "45지역 · 원본 270비행 · 추가 학습 180비행 · 새 난수 24비행. 해시·공간 분리·완주 검사 통과",
        "bank_max_difference": bank["maximum_feature_difference"],
        "checkpoint_sha256": modelsha,
        "calibration_sha256": calsha,
        "policy_sha256": policysha,
        "runtime": rt,
        "replay": replay,
        "learned_repositioning_implemented": False,
        "raw_integrity_sha256": sha256_file(RESULT / "raw_integrity.json"),
    }
    write_json(RESULT / "final_audit.json", out)
    print(
        json.dumps(
            {
                "pass": True,
                "tests_passed": tests.testsRun,
                "maps": 45,
                "flights": 474,
                "full_replay_endpoints": replay["full_endpoints"],
            }
        )
    )


if __name__ == "__main__":
    main()
