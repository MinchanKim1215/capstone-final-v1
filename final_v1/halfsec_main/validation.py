"""Run the released controller on known stalls and on disjoint map cases."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptive_joint_expanded.contracts import lock
from fast_halfsec import evaluation
from fast_halfsec.protocol import KINDS
from forecast_rebuild.contracts import digest, now, read, sha, write

from final_v1 import ARTIFACTS
from . import ROOT
from .release import EXPANDED_REGISTRY, MANIFEST, run


RUN_DIR = ARTIFACTS / 'halfsec_main'
PLAN, STATUS, RESULTS = (RUN_DIR / f'{name}.json' for name in ('protocol', 'status', 'results'))
EXPANDED = EXPANDED_REGISTRY.parent
REPLICATION = ARTIFACTS / 'relocation/replication'


def prepare():
    confirmation = [read(path) for path in (EXPANDED / 'flights/confirmation').glob('*.json')]
    stalled = [row for row in confirmation if row['arm'] == 'joint_05s' and not row['mission_completed']]
    if len(stalled) != 6:
        raise ValueError('Expected six archived incomplete half-second flights')
    replicas = {evaluation.key(row): row for row in
                (read(path) for path in (REPLICATION / 'baseline').glob('*.json'))}
    replica_plan = read(REPLICATION / 'protocol.json')
    regions = sorted({spec['region'] for spec in replica_plan['cases']})
    choose = {(region, KINDS[index % len(KINDS)]) for index, region in enumerate(regions)}
    independent = [row for row in replicas.values() if (row['spec']['region'], row['spec']['kind']) in choose]
    if len(independent) != len(regions) or len(independent) != 8:
        raise ValueError('Independent comparison requires all eight archived map baselines')
    registry = {entry['key']: entry for entry in read(EXPANDED_REGISTRY)['maps']}
    groups = {registry[row['spec']['region']]['study_group'] for row in confirmation}
    if any(registry[row['spec']['region']]['study_group'] in groups for row in independent):
        raise ValueError('The independent maps share geography with the original evaluation')
    references = {'known_stalls': stalled, 'separate_maps': independent}
    plan = dict(
        checkpoint=read(MANIFEST), registry_sha256=sha(EXPANDED_REGISTRY),
        sources={filename: sha(ROOT / filename) for filename in
                 ('__init__.py', 'mission.py', 'navigation.py', 'release.py', 'validation.py')},
        cases={name: [{'spec': row['spec'], 'reference_sha256': row['archive_sha256']}
                      for row in sorted(rows, key=evaluation.key)] for name, rows in references.items()},
    )
    if PLAN.is_file() and read(PLAN) != plan:
        raise ValueError('This validation uses another frozen policy or set of flights')
    if not PLAN.is_file():
        write(PLAN, plan)
    return references, registry


def simulate(spec, protocol):
    folder = RUN_DIR / 'flights'
    identifier = digest(spec)[:24]
    meta, archive = folder / (identifier + '.json'), folder / (identifier + '.npz')
    if meta.is_file():
        row = read(meta)
        if row['protocol'] != protocol or sha(archive) != row['archive_sha256']:
            raise ValueError('Saved flight does not match the frozen validation')
        return row
    from final_v1.halfsec_joint.run import retry_permission

    row, arrays = retry_permission(run, spec)
    folder.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, links=arrays['links'], network=arrays['network'])
    retry_permission(Path.replace, pending, archive)
    row.update(protocol=protocol, archive_path=str(archive), archive_sha256=sha(archive))
    write(meta, row)
    print(f"{spec['region']} {spec['kind']}: {row['duration_s']:.1f}s, {row['onsets']} cuts, "
          f"complete={row['mission_completed']}, clearance={row['cleared_stages']}, "
          f"recovery filters={row['recovery_filtered']}, {row['wall_s']:.1f}s wall", flush=True)
    return row


def summarize(references, flights, registry, protocol):
    candidate = {evaluation.key(row): row for row in flights}
    comparisons = {}
    with patch.object(evaluation, 'registry', return_value=registry):
        for name, before in references.items():
            before.sort(key=evaluation.key)
            after = [candidate[evaluation.key(row)] for row in before]
            routes = [dict(spec=row['spec'], completed=row['mission_completed'], duration_s=row['duration_s'],
                           waypoints_reached=row['waypoints_reached'], waypoints_total=row['waypoints_total'],
                           onsets=row['onsets'], cut_seconds=row['cut_seconds'],
                           cleared_stages=row['cleared_stages']) for row in after]
            comparisons[name] = dict(before=evaluation.pooled(before), after=evaluation.pooled(after),
                                     paired=evaluation.paired(before, after), audit=evaluation.audit(before + after),
                                     cleared_stages=sum(row['cleared_stages'] for row in after),
                                     recovery_filtered=sum(row['recovery_filtered'] for row in after),
                                     routes=routes)
    return dict(completed_at=now(), protocol_sha256=protocol, **comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with lock(RUN_DIR / 'run.lock'):
        references, registry = prepare()
        specs = [row['spec'] for rows in references.values() for row in rows]
        protocol = sha(PLAN)
        flights = []
        write(STATUS, dict(state='running', completed=0, total=len(specs), updated_at=now()))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            tasks = [pool.submit(simulate, spec, protocol) for spec in specs]
            for task in as_completed(tasks):
                flights.append(task.result())
                write(STATUS, dict(state='running', completed=len(flights), total=len(specs), updated_at=now()))
        write(RESULTS, summarize(references, flights, registry, protocol))
        write(STATUS, dict(state='completed', completed=len(flights), total=len(specs), updated_at=now()))


if __name__ == '__main__':
    main()
