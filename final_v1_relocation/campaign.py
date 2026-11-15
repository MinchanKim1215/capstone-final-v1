"""개선안 비교 비행을 실행하고 results.json과 RESULTS.md를 만든다.

* 확인 비행: Final V1 protocol.json의 confirmation 42개(지도 14 × 간섭 3) × options.ARMS
* 명령 손실: Final V1 protocol.json의 stress 14조건 × options.STRESS_ARMS
같은 지도·난수에서 V21(모든 개선안 끔)을 다시 비행해 기준으로 쓴다. 중간에 멈춰도 다시 실행하면
완료된 비행은 건너뛴다(소스가 바뀌면 다시 비행).

예)
  python -m final_v1_relocation.campaign                       # 전체
  python -m final_v1_relocation.campaign --arms v21,r1_cut_recovery --skip-stress
  python -m final_v1_relocation.campaign --limit 2 --steps 40  # 짧은 동작 확인
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import traceback

import numpy as np

from final_v1 import ARTIFACTS as FINAL
from forecast_rebuild.contracts import now, read, sha, write

from . import ARTIFACTS, ROOT, VERSION
from .evaluation import PRIMARY, adoptable, compare, describe, fallback_reasons, select_evidence, stable_recovery
from .options import ARMS, DEV_ARMS, DEV_MAPS, DEV_SEEDS, FALLBACKS, STRESS_ARMS
from .runner import CONTROLLERS, initialize, run

SOURCES = ("__init__.py", "options.py", "graph.py", "recovery.py", "bypass.py", "control.py", "transport.py",
           "runner.py")


def pins():
    return {name: sha(ROOT/name) for name in SOURCES}


def specs():
    path = FINAL/"protocol.json"
    if path.exists():
        plan = read(path)
        return plan["confirmation"], plan["stress"]
    return regenerate()


def regenerate():
    """protocol.json이 없으면 final_v1.protocol.prepare()와 같은 규칙으로 조건을 다시 만든다.

    확인 지도는 expanded_plan.json의 검증 지도 공간 분할(folds)에서, 난수와 방향은 Final V1 버전 문자열의
    해시에서 나온다. 재생 HTML에 저장된 56개 조건과 지도·간섭·방향이 모두 같은지 확인한다.
    """
    from final_v1.protocol import KINDS, make_spec
    from forecast_rebuild.contracts import folds

    try:
        regions = folds()["confirmation"]
    except FileNotFoundError as error:
        raise SystemExit(f"조건을 만들 지도 목록이 없습니다: {error.filename}\n"
                         "additional_maps_20261030/expanded_plan.json과 확인 지도 14개가 필요합니다. "
                         "자료 없이 확인할 수 있는 것은 unittest와 replay_audit입니다.")
    confirm = [make_spec(region, "confirm", kind) for region in regions for kind in KINDS]
    stress = [make_spec(region, "stress", KINDS[index % 3]) for index, region in enumerate(regions)]
    check_against_replay(confirm, stress)
    return confirm, stress


def check_against_replay(confirm, stress):
    from .replay_audit import REPLAY, load

    if not REPLAY.exists():
        return
    cases = load()["cases"]
    recorded = {phase: sorted((c["region"], c["kind"], c["direction"]) for c in cases if c["phase"] == phase)
                for phase in ("비교", "명령 손실")}
    made = {"비교": sorted((s["region"], s["kind"], s["direction"]) for s in confirm),
            "명령 손실": sorted((s["region"], s["kind"], s["direction"]) for s in stress)}
    if made != recorded:
        raise ValueError("다시 만든 조건이 Final V1 재생 기록과 다릅니다. final_v1/artifacts/protocol.json이 필요합니다.")


def name(spec, stage="confirm"):
    base = f'{spec["region"]}_{spec["kind"]}'
    return base if not stage.startswith("dev") else f'{base}_{spec["seed"]}'


def dev_specs(round_=0):
    """확인 비행과 겹치지 않는 학습 지도의 새 난수 조건. round_ > 0 이면 앞 단계와 겹치지 않는 새 난수."""
    from final_v1.protocol import KINDS, make_spec

    token = "relocation-dev" if not round_ else f"relocation-dev{round_}"
    return [make_spec(region, f"{token}-{i}", kind)
            for region in DEV_MAPS for kind in KINDS for i in range(DEV_SEEDS)]


def folder(controller, steps):
    return ARTIFACTS/(controller if steps is None else f"{controller}_steps{steps}")


def trial(task):
    spec, arm, transport, controller, steps, sources = task[:6]
    stage = task[6] if len(task) > 6 else ("confirm" if transport is None else "stress")
    label = arm if transport is None else f"{arm}__{transport}"
    path = folder(controller, steps)/stage/label/(name(spec, stage)+".json")
    array = path.with_suffix(".npz")
    contract = dict(spec=spec, arm=arm, transport=transport, controller=controller, steps=steps, sources=sources)
    if path.exists() and array.exists():
        row = read(path)
        if row.get("contract") == contract and sha(array) == row["archive_sha256"]:
            return row
    row, data = run(spec, arm, transport=transport, max_steps=steps, controller=controller)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = array.with_suffix(".pending.npz")
    np.savez_compressed(pending, **data)
    pending.replace(array)
    row.update(contract=contract, archive_path=str(array), archive_sha256=sha(array))
    write(path, row)
    return row


def simulate(stage, tasks, workers, controller):
    records = [None]*len(tasks)
    if workers <= 1:
        initialize(controller)
        for i, task in enumerate(tasks):
            records[i] = trial(task)
            print(stage, i+1, "/", len(tasks), flush=True)
        return records
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize,
                             initargs=(controller,)) as pool:
        jobs = {pool.submit(trial, task): i for i, task in enumerate(tasks)}
        for count, job in enumerate(as_completed(jobs), 1):
            i = jobs[job]
            try:
                records[i] = job.result()
            except Exception as error:
                for pending in jobs:
                    pending.cancel()
                raise RuntimeError(f"Flight failed: {tasks[i][:4]}") from error
            if count % 6 == 0 or count == len(tasks):
                print(stage, count, "/", len(tasks), flush=True)
    return records


def archive(row):
    with np.load(row["archive_path"], allow_pickle=False) as data:
        return {k: data[k] for k in ("network", "fallback", "received") if k in data.files}


def consistency(rows):
    """모든 개선안을 끈 v21이 Final V1에 저장된 V21 비행과 같은지 확인한다."""
    same, checked = 0, 0
    for row in rows:
        old = FINAL/"confirm"/"v21"/(name(row["spec"])+".json")
        if not old.exists():
            continue
        prior = read(old)
        checked += 1
        same += (prior["persistent_onsets"] == row["persistent_onsets"]
                 and prior["persistent_cut_intervals"] == row["persistent_cut_intervals"])
    return dict(checked=checked, identical=same)


def patterns(grouped):
    """급락형·점진형·혼합형 단절 수 (Final V1의 원인 분석 재사용). 실패하면 이유만 남긴다."""
    try:
        from final_v1.campaign import native_patterns
        from final_v1.protocol import maps

        from .paths import localize_registry

        registry = localize_registry(maps())
        result = {}
        for arm in ("v21", "r4_bypass", "r13_combined"):
            if arm not in grouped:
                continue
            counts = defaultdict(int)
            for row in grouped[arm]:
                for start, _, shape in native_patterns(row, registry):
                    if start > 0.:
                        counts[shape] += 1
            result[arm] = dict(counts)
        return result
    except Exception as error:  # 지도 원인 자료가 없는 환경
        return dict(error=repr(error))


def group_lookup():
    from final_v1.protocol import maps

    registry = maps()
    return lambda row: registry[row["spec"]["region"]]["joint_spatial_group"]


def evaluate_confirm(rows, flights):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["relocation_arm"]].append(row)
    group_of = group_lookup()
    stats = {arm: describe(items) for arm, items in grouped.items()}
    versus = {arm: compare(grouped["v21"], items, group_of) for arm, items in grouped.items() if arm != "v21"}
    adoption = {arm: adoptable(stats[arm], change, flights, PRIMARY.get(arm, "starts_per_h"))
                for arm, change in versus.items()}
    return grouped, dict(stats=stats, vs_v21=versus, adoptable=adoption,
                         primary={arm: PRIMARY.get(arm, "starts_per_h") for arm in versus})


def evaluate_stress(rows, flights):
    grouped = defaultdict(list)
    for row in rows:
        grouped[f'{row["relocation_arm"]}__{row["transport"]}'].append(row)
    group_of = group_lookup()
    reference = grouped["v21__neutral"]
    stats = {label: dict(describe(items), **stable_recovery(items, archive),
                         fallback=fallback_reasons(items, archive)) for label, items in grouped.items()}
    versus = {label: compare(reference, items, group_of) for label, items in grouped.items()
              if label != "v21__neutral"}
    return dict(stats=stats, vs_neutral=versus, flights=flights)


def run_dev(args):
    specs = dev_specs(args.dev_round)
    if args.limit:
        specs = specs[:args.limit]
    arms = DEV_ARMS if not args.dev_round else tuple(dict.fromkeys(["v21"]+[a for a in args.arms.split(",") if a]))
    stage = "dev" if not args.dev_round else f"dev{args.dev_round}"
    suffix = "" if not args.dev_round else f"_round{args.dev_round}"
    sources = pins()
    out = folder(args.controller, args.steps)
    rows = simulate(stage, [(spec, arm, None, args.controller, args.steps, sources, stage)
                            for spec in specs for arm in arms], args.workers, args.controller)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["relocation_arm"]].append(row)
    group_of = group_lookup()
    stats = {arm: describe(items) for arm, items in grouped.items()}
    versus = {arm: compare(grouped["v21"], items, group_of) for arm, items in grouped.items() if arm != "v21"}
    result = dict(version=VERSION, stage=stage, maps=list(DEV_MAPS), seeds=DEV_SEEDS, flights=len(specs),
                  sources=sources, stats=stats, vs_v21=versus, selection=select_evidence(stats, arms),
                  finished_at=now())
    write(out/f"dev_results{suffix}.json", result)
    from .report import dev_markdown

    (out/f"DEV_RESULTS{suffix}.md").write_text(dev_markdown(result), encoding="utf-8")
    print("개발 결과:", out/f"DEV_RESULTS{suffix}.md", "선택:", result["selection"]["selected"], flush=True)


def main():
    parser = argparse.ArgumentParser(description="Final V1 재배치 개선안 비교")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--stress-arms", default=",".join(f"{a}:{f}" for a, f in STRESS_ARMS))
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--controller", choices=CONTROLLERS, default="v21")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2)-1)))
    parser.add_argument("--limit", type=int, help="앞의 N개 조건만 (동작 확인용)")
    parser.add_argument("--steps", type=int, help="비행을 N 명령 주기에서 끊는다 (동작 확인용)")
    parser.add_argument("--dev", action="store_true",
                        help="복구 근거 선택: 학습 지도 5곳 × 간섭 3 × 난수 3에서 DEV_ARMS 비교")
    parser.add_argument("--dev-round", type=int, default=0, help="개발 비행의 새 난수 묶음 번호 (0 = 선택 단계)")
    args = parser.parse_args()
    if args.dev:
        return run_dev(args)
    arms = list(dict.fromkeys(["v21"]+[a for a in args.arms.split(",") if a]))
    stress_arms = [tuple(item.split(":")) for item in args.stress_arms.split(",") if item]
    if ("v21", "neutral") not in stress_arms:
        stress_arms.insert(0, ("v21", "neutral"))
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"알 수 없는 설정: {arm}")
    for arm, fallback in stress_arms:
        if arm not in ARMS or fallback not in FALLBACKS:
            parser.error(f"알 수 없는 명령 손실 설정: {arm}:{fallback}")
    confirm, stress = specs()
    if args.limit:
        confirm, stress = confirm[:args.limit], stress[:args.limit]
    sources = pins()
    out = folder(args.controller, args.steps)
    result = dict(version=VERSION, controller=args.controller, steps=args.steps, limit=args.limit,
                  sources=sources, final_v1_protocol_sha256=sha(FINAL/"protocol.json"), started_at=now())
    if not args.skip_confirm:
        rows = simulate("confirm", [(spec, arm, None, args.controller, args.steps, sources)
                                    for spec in confirm for arm in arms], args.workers, args.controller)
        grouped, result["confirmation"] = evaluate_confirm(rows, len(confirm))
        result["pattern_counts"] = patterns(grouped)
        if args.controller == "v21" and args.steps is None:
            result["v21_matches_final_v1"] = consistency(grouped["v21"])
    if not args.skip_stress:
        rows = simulate("stress", [(spec, arm, fallback, args.controller, args.steps, sources)
                                   for spec in stress for arm, fallback in stress_arms], args.workers, args.controller)
        result["command_loss"] = evaluate_stress(rows, len(stress))
    result["finished_at"] = now()
    write(out/"results.json", result)
    from .report import markdown

    (out/"RESULTS.md").write_text(markdown(result), encoding="utf-8")
    print("결과:", out/"RESULTS.md", flush=True)


if __name__ == "__main__":
    keep_awake = os.name == "nt"
    if keep_awake:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
