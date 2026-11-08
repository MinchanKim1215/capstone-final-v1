"""Read-only model evaluations outside the frozen training packages."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import adaptive_joint_expanded as package
from adaptive_joint_expanded.contracts import read_json, atomic_json, sha, digest
from adaptive_joint_expanded.experiment import assert_protocol

RUN = package.ARTIFACTS / 'runs/adaptive_v7_expanded100_20261031'
SOURCE = RUN / 'validation/u0000064'
OUT = ROOT / 'training_ops/reports/expanded_results_20261101'
INITIAL = RUN / 'checkpoints/s0000000.pt'
BEST = RUN / 'checkpoints/s0000064.pt'


def verify_initial():
    import torch
    binding = read_json(package.ROOT / 'configs/parent_initialization.json')
    parent = Path(binding['checkpoint'])
    assert sha(parent) == binding['checkpoint_sha256']
    before = torch.load(parent, map_location='cpu', weights_only=False)
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
    assert before['model'].keys() == initial['model'].keys()
    assert all(torch.equal(value, initial['model'][key]) for key, value in before['model'].items())
    return dict(parent_checkpoint=str(parent), parent_sha256=sha(parent),
                initial_checkpoint=str(INITIAL), initial_sha256=sha(INITIAL),
                all_model_tensors_identical=True)


def communication(protocol):
    from adaptive_joint_expanded.driver import evaluate
    metadata = [read_json(path) for path in sorted((SOURCE / 'rollouts').glob('*.json'))]
    specs = [row['spec'] for row in metadata]
    identity = verify_initial()
    result = evaluate(INITIAL, protocol, OUT / 'before_policy', explicit_specs=specs)
    after = read_json(SOURCE / 'summary.json')
    baseline_rows = [read_json(path) for path in sorted((OUT / 'before_policy/rollouts').glob('*.json'))]
    assert [row['spec'] for row in baseline_rows] == specs
    assert len(metadata) == 8 and result['completed'] == 8
    rows = []
    for old, new in zip(baseline_rows, metadata):
        assert old['spec'] == new['spec']
        rows.append(dict(region=new['region'], spec=new['spec'],
                         before_deployment=old['deployment'], after_deployment=new['deployment'],
                         before_sha256=old['array_sha256'], after_sha256=new['array_sha256'],
                         before={key:old[key] for key in result['means']},
                         after={key:new[key] for key in result['means']}))
    output = dict(scope='Same 8 development map/direction/interference/seed scenarios; separate adaptive-policy flights. Not sealed final test.',
                  parent_identity=identity, before=result, after=after, cases=rows,
                  protocol_sha256=digest(protocol),
                  caution='Development selection cases. Results include extra training, new data and changed policy; map addition alone is not isolated.')
    atomic_json(OUT / 'communication.json', output)
    print(json.dumps(dict(kind='communication', before=result['means'], after=after['means']), ensure_ascii=False), flush=True)


def prediction(protocol):
    import numpy as np
    import torch
    from adaptive_joint_expanded.model import Controller
    from adaptive_joint_expanded.targets import timing_labels, binary_targets, probabilities, TARGET_NAMES
    from adaptive_joint_expanded.metrics import alert_counts, summarize_alerts
    from solo.metrics import binary_metrics
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    identity = verify_initial()
    models = {}
    for name, path in [('before', INITIAL), ('after', BEST)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['protocol_sha256'] == digest(protocol)
        model = Controller(warm_start=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        models[name] = model.to(device).eval()
    plan = read_json(ROOT / 'additional_maps_20261030/expanded_plan.json')
    added = {row['key'] for row in plan['maps'] if row.get('source_dataset', 'legacy_v6') != 'legacy_v6'}
    truth = []
    scores = {name:[] for name in models}
    alarm_results = {name:[] for name in models}
    case_results = []
    maximum_reproduction_difference = 0.
    for meta in sorted((SOURCE / 'rollouts').glob('*.json')):
        row = read_json(meta)
        assert not row['smoke_only']
        assert sha(meta.with_suffix('.npz')) == row['array_sha256']
        keys = ('links','rf_values','rf_masks','rf_intents','aux_endpoints','aux_geometry',
                'aux_labels','aux_valid','aux_support','aux_probabilities')
        with np.load(meta.with_suffix('.npz'), allow_pickle=False) as archive:
            a = {key:archive[key] for key in keys}
        y, valid = timing_labels(a['links'], a['aux_endpoints'], row['search_drones'])
        assert np.array_equal(y, a['aux_labels']) and np.array_equal(valid, a['aux_valid'])
        prediction = {name:np.empty_like(a['aux_probabilities']) for name in models}
        for start in range(0, len(y), 16):
            stop = min(start + 16, len(y))
            ix = a['aux_endpoints'][start:stop,None] + np.arange(-40, 0)[None,:]
            v = torch.as_tensor(a['rf_values'][ix].transpose(0,2,3,1,4).copy(), dtype=torch.float32, device=device)
            m = torch.as_tensor(a['rf_masks'][ix].transpose(0,2,3,1,4).copy(), device=device)
            c = torch.as_tensor(a['rf_intents'][ix].transpose(0,2,3,1,4).copy(), dtype=torch.float32, device=device)
            g = torch.as_tensor(a['aux_geometry'][start:stop], device=device)
            with torch.inference_mode():
                for name, model in models.items():
                    _, timing = model.forecaster.encode(v,m,c,g)
                    prediction[name][start:stop] = probabilities(timing).cpu().numpy()
        difference = float(np.max(np.abs(prediction['after'] - a['aux_probabilities'])))
        maximum_reproduction_difference = max(maximum_reproduction_difference, difference)
        assert difference < 2e-5, difference
        good = valid & a['aux_support']
        labels = binary_targets(y)[good]
        truth.append(labels)
        per_case = {}
        for name in models:
            values = prediction[name][good]
            scores[name].append(values)
            alarms = alert_counts(dict(a, aux_probabilities=prediction[name]), row['search_drones'])
            alarm_results[name].append(alarms)
            per_case[name] = dict(next5=binary_metrics(labels[:,0], values[:,0]), alarms=summarize_alerts([alarms]))
        case_results.append(dict(region=row['region'], source_group='added' if row['region'] in added else 'original',
                                 source_array_sha256=row['array_sha256'], samples=len(labels), metrics=per_case))
        print(json.dumps(dict(kind='prediction', completed=len(case_results), cases=8, region=row['region'])), flush=True)
    labels = np.concatenate(truth)
    result = {}
    for name in models:
        values = np.concatenate(scores[name])
        result[name] = {target:binary_metrics(labels[:,h], values[:,h]) for h,target in enumerate(TARGET_NAMES)}
        result[name]['alarms_1s'] = summarize_alerts(alarm_results[name])
    output = dict(scope='Same archived RF histories, causal masks, issued intents, geometry, support, timestamps and targets. 8 development flights, not final test.',
                  common_samples=len(labels), parent_identity=identity, models=result, cases=case_results,
                  after_checkpoint=str(BEST), after_sha256=sha(BEST), protocol_sha256=digest(protocol),
                  recorded_after_probability_max_abs_difference=maximum_reproduction_difference,
                  evaluation_device=device,
                  caution='Trajectory generated by selected v7 step64 policy. Development selection cases; not isolated map-addition effect or independent final generalization.')
    atomic_json(OUT / 'prediction.json', output)
    print(json.dumps(dict(kind='prediction', common_samples=len(labels), models=result),ensure_ascii=False), flush=True)


def main():
    mode = argparse.ArgumentParser()
    mode.add_argument('kind', choices=['communication','prediction'])
    args = mode.parse_args()
    protocol = read_json(package.ARTIFACTS / 'protocol.json')
    assert_protocol(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    (communication if args.kind == 'communication' else prediction)(protocol)


if __name__ == '__main__':
    main()
