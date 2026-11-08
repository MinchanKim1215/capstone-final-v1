"""설치 후 사용할 하나의 단계별 실행 진입점입니다."""
import argparse
import json

from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.map_setup import digest


def validate_frozen_protocol():
    protocol = read_json(ROOT/'configs'/'upgrade_protocol.json')
    expected = protocol['protocol_sha256']
    if digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}) != expected:
        raise ValueError('동결한 실험 설정의 내용과 해시가 다릅니다.')
    for entry in protocol['maps']:
        if sha256_file(ROOT/entry['path']) != entry['sha256']:
            raise ValueError(f'{entry["key"]}: 지도 해시가 다릅니다.')
    return protocol


def collect(workers):
    from upgrade.collect import collect_all
    result = collect_all(workers=workers)
    if not result['complete'] or len(result['records']) != 270:
        raise RuntimeError('전체 270개 실행을 완료하지 못했습니다. 실패 목록을 확인하십시오.')
    return result


def evaluate():
    from upgrade.evaluate import evaluate_banks, evaluate_all_dense
    from upgrade.operational import run
    evaluate_banks()
    evaluate_all_dense()
    operational = run()
    from upgrade.diagnostics import run as diagnose
    diagnose()
    from upgrade.motion_diagnostics import run as diagnose_motion
    diagnose_motion()
    return operational


def main():
    parser = argparse.ArgumentParser(description='45개 지도 Transformer 개선 실험')
    parser.add_argument('stage', choices=['collect', 'audit', 'prepare', 'train', 'evaluate', 'report', 'all'])
    parser.add_argument('--workers', type=int, choices=[1,2,3,4], default=2)
    args = parser.parse_args()
    validate_frozen_protocol()
    stages = ['collect','audit','prepare','train','evaluate','report'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print(json.dumps({'stage': stage, 'status': 'start'}, ensure_ascii=False), flush=True)
        if stage == 'collect':
            collect(args.workers)
        elif stage == 'audit':
            from upgrade.audit import run
            run()
            from upgrade.motion_diagnostics import run as diagnose_motion
            diagnose_motion()
        elif stage == 'prepare':
            from upgrade.data import prepare
            prepare()
        elif stage == 'train':
            from upgrade.pipeline import run
            run(min(args.workers, 2), stop_after_training=True)
        elif stage == 'evaluate':
            evaluate()
        elif stage == 'report':
            from upgrade.report import create
            create()
        print(json.dumps({'stage': stage, 'status': 'complete'}, ensure_ascii=False), flush=True)
    if args.stage == 'all':
        write_json(ROOT/'results'/'upgrade'/'progress.json', {'stage': 'report_complete', 'complete': True})


if __name__ == '__main__':
    main()
