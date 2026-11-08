"""고정된 검증 실행만으로 운용 방식·최선 학습 가중치 선택."""

import argparse, json, shutil
from swarm_sim.regions import ROOT, sha256_file
from cooperative.analyze import validation_select


def choose(candidates):
    result = validation_select(candidates)
    learned = [
        k for k in result["ordered_feasible"] if candidates[k]["controller"] == "ppo"
    ]
    if not learned:
        learned = sorted(
            (k for k in candidates if candidates[k]["controller"] == "ppo"),
            key=lambda k: 1 - result["candidates"][k]["macro"]["search_connected_mean"],
        )
    if learned:
        label = learned[0]
        source = candidates[label]["checkpoint"]
        dest = ROOT / "model/cooperative/best_learned.pt"
        shutil.copy2(ROOT / source, dest)
        result["best_learned"] = {
            "label": label,
            "source_checkpoint": source,
            "checkpoint": dest.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(dest),
        }
    file = ROOT / "results/cooperative/selection.json"
    tmp = file.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(file)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="configs/cooperative_candidates.json")
    args = parser.parse_args()
    if (ROOT / "configs/cooperative_test_freeze.json").exists():
        raise RuntimeError(
            "최종 시험까지 완료한 선정 파일 보호. 새 실험은 프로젝트 복사본에서 freeze 파일·기존 평가 결과를 별도로 관리한 뒤 실행 필요."
        )
    data = json.loads((ROOT / args.candidates).read_text(encoding="utf-8"))
    result = choose(data)
    print(
        json.dumps(
            {
                "selected": result["selected"],
                "best_learned": result.get("best_learned"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
