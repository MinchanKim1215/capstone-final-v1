"""Assemble the team handoff from the frozen local research records."""
from pathlib import Path
from collections import Counter
import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/final_v1_team_20261107'
CODE = OUT / 'code'
TITLE = '캡스톤 디자인 II · Final V1 통합보고서'
REPORT_NAME = 'Final_V1_통합보고서.html'
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'node_modules', 'artifacts', 'results',
             'cache', 'runs', 'maps', 'model', 'raw', 'data', '.sites-runtime'}
SOURCE_EXTENSIONS = {'.py', '.cjs', '.js', '.css', '.html', '.md', '.json', '.txt',
                     '.bat', '.ps1', '.yaml', '.yml', '.toml'}
COMPLETE_ARTIFACTS = ('final_v1', 'forecast_learning', 'forecast_shield',
                      'forecast_avoidance', 'forecast_prevention', 'outage_atlas',
                      'final_v1_margin', 'final_v1_fast')
PACKAGES = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and
                  (p.name.startswith(('forecast_', 'final_v1', 'adaptive_joint')) or
                   p.name in ('drone_forecast_joint_training', 'prediction_improvement', 'outage_atlas')))


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def walk_files(folder, skip=()):
    for parent, directories, names in os.walk(folder):
        directories[:] = sorted(name for name in directories if name not in skip)
        for name in sorted(names):
            yield Path(parent) / name


def build_code():
    for name in PACKAGES:
        folder = ROOT / name
        for source in walk_files(folder, SKIP_DIRS):
            if source.suffix in SOURCE_EXTENSIONS and source.stat().st_size < 1_000_000:
                copy_file(source, CODE / source.relative_to(ROOT))
        artifacts = folder / 'artifacts'
        for source in artifacts.glob('*.json'):
            if not any(word in source.name for word in ('publication', 'approval', 'access')):
                copy_file(source, CODE / source.relative_to(ROOT))
        for source in (artifacts / 'models').glob('*'):
            if source.suffix in ('.pt', '.txt', '.json'):
                copy_file(source, CODE / source.relative_to(ROOT))
    for name in COMPLETE_ARTIFACTS:
        for source in walk_files(ROOT / name / 'artifacts', {'numba_cache', '__pycache__', 'viewer_qa'}):
            if source.suffix in ('.json', '.npz', '.pt', '.txt', '.png') and not any(
                    word in source.name for word in ('publication', 'approval', 'access')):
                copy_file(source, CODE / source.relative_to(ROOT))
    plan_path = Path('additional_maps_20261030/expanded_plan.json')
    copy_file(ROOT / plan_path, CODE / plan_path)
    plan = read(plan_path)
    map_rows = []
    for entry in plan['maps']:
        source = ROOT / 'drone_forecast_joint_training' / entry['path']
        source = source.resolve()
        relative = source.relative_to(ROOT)
        if sha(source) != entry['sha256']:
            raise ValueError(f"Map content changed: {entry['key']}")
        copy_file(source, CODE / relative)
        map_rows.append(dict(key=entry['key'], path=relative.as_posix(), sha256=entry['sha256'],
                             split=entry['joint_split']))
    selected_files = (
        'adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031/checkpoints/s0000064.pt',
        'adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031/best.json',
        'adaptive_joint/artifacts/runs/adaptive_v6_20261030/checkpoints/s0000192.pt',
        'adaptive_joint/artifacts/runs/adaptive_v6_20261030/status.json',
        'prediction_improvement/runs/next_outage_1s_v1/best.pt',
        'drone_forecast_joint_training/results/transformer/train/transformer_main/checkpoints/s0000176.pt',
    )
    for name in selected_files:
        copy_file(ROOT / name, CODE / name)
    for name in ('advance/selected', 'continuity', 'advance/transformer_reference'):
        for source in walk_files(ROOT / 'drone_forecast_joint_training/model' / name):
            if source.suffix in ('.pt', '.json', '.txt') and 'checkpoints' not in source.parts:
                copy_file(source, CODE / source.relative_to(ROOT))
    for source in (ROOT / 'drone_forecast_joint_training/configs').rglob('*.json'):
        copy_file(source, CODE / source.relative_to(ROOT))
    for source in (ROOT / 'training_ops').glob('*.py'):
        if not any(word in source.name for word in ('download', 'run_training', 'queue')):
            copy_file(source, CODE / source.relative_to(ROOT))
    for source in (ROOT / 'training_ops/reports').glob('*.md'):
        copy_file(source, CODE / source.relative_to(ROOT))
    for name in ('analyze_prior_drops_20261105.py', 'precursor_fine_controls_20261105.py',
                 'extend_outage_history_20261105.py', 'build_outage_patterns_20261105.py',
                 'build_long_history_20261105.py', 'build_precursor_report_20261105.py'):
        copy_file(ROOT / 'tmp' / name, CODE / 'tmp' / name)
    copy_file(ROOT / 'tmp/team_runtime_20261107.py', CODE / 'run_final_v1.py')
    copy_file(ROOT / 'tmp/team_portability_20261107.py', CODE / 'team_portability.py')
    write_json(CODE / 'handoff_paths.json', {'source_root': str(ROOT), 'maps': map_rows})
    write_json(OUT / 'evidence/included_maps.json', {'maps': map_rows, 'count': len(map_rows)})
    print('Code and 345 verified maps copied', flush=True)


def collect_metrics():
    comparison = read('final_v1/artifacts/comparison.json')
    shield = read('forecast_shield/artifacts/final_results.json')
    release = read('final_v1/artifacts/release.json')
    causes = read('final_v1/artifacts/causes.json')
    a, b = comparison['alerts']['v20'], comparison['alerts']['v21']
    old, new = comparison['stats']['v7'], comparison['stats']['v21']
    metrics = dict(
        title=TITLE, assembled_on='2026-11-07', results_as_of='2026-11-05',
        primary_cohort=dict(maps=14, flights=42, spatial_groups=12, hours=a['hours'],
                            events=a['events'], ranking_rows=a['ranking_rows'],
                            ranking_positives=a['ranking_positives'], lead_window_s=[3, 7]),
        prediction=comparison['alerts'], control=comparison['stats'],
        prediction_change=dict(recall_pp=100 * (b['recall'] - a['recall']),
                               precision_pp=100 * (b['precision'] - a['precision']),
                               ap_pp=100 * (b['average_precision'] - a['average_precision']),
                               false_or_late_delta=b['false_or_late_per_fleet_h'] - a['false_or_late_per_fleet_h'],
                               false_or_late_reduction_pct=100 * (1 - b['false_or_late'] / a['false_or_late'])),
        control_change=dict(starts_reduction_pct=100 * (1 - new['starts_per_h'] / old['starts_per_h']),
                            cut_time_reduction_pct=100 * (1 - new['cut_s_per_h'] / old['cut_s_per_h']),
                            timely_pdr_pp=100 * (new['timely_pdr'] - old['timely_pdr']),
                            relay_m_delta=new['mean_relay_m'] - old['mean_relay_m'],
                            rate_difference_ci95=[-x for x in reversed(comparison['vs_v21']['v7']['ci95'])]),
        earlier_same_cohort=dict(v18=shield['v18'], v20=shield['priority'],
                                 change_pp=100 * (shield['priority']['recall'] - shield['v18']['recall'])),
        causes={k: v for k, v in causes.items() if k != 'flights'},
        patterns_in_confirmation=comparison['pattern_counts'],
        joint_control=comparison['continuous_vs_matched_ppo'],
        release={key: release[key] for key in ('version', 'controller', 'transport_fallback',
                                              'confirmation_flights', 'stress_flights', 'replay_cases')},
        thresholds=read('final_v1/artifacts/comparison_freeze.json')['thresholds'],
        operational_score=dict(learned_weight=.7, physical_weight=.3,
                               threshold=read('forecast_learning/artifacts/freeze.json')['chosen']['threshold']),
    )
    assert a['events'] == b['events'] == 246
    assert a['ranking_rows'] == b['ranking_rows'] == 25079
    assert a['ranking_positives'] == b['ranking_positives'] == 211
    assert sum(row['events'] for row in causes['by_pattern'].values()) == 2572
    assert old['starts'] == 246 and new['starts'] == 166
    write_json(OUT / 'evidence/report_metrics.json', metrics)
    sources = ('final_v1/artifacts/comparison.json', 'final_v1/artifacts/comparison_freeze.json',
               'final_v1/artifacts/causes.json', 'final_v1/artifacts/training.json',
               'final_v1/artifacts/release.json', 'final_v1/artifacts/completed.json',
               'forecast_learning/artifacts/final_results.json', 'forecast_shield/artifacts/final_results.json',
               'output/outage_patterns_20261105/evidence.json',
               'output/outage_history_20261105/long_history.json',
               'output/precursor_drops_20261105/summary.json',
               'output/precursor_drops_20261105/fine_controls.json',
               'output/precursor_drops_20261105/report_evidence.json',
               'final_v1_fast/artifacts/results.json', 'final_v1_margin/artifacts/results.json')
    for name in sources:
        copy_file(ROOT / name, OUT / 'evidence/original' / name)
    write_json(OUT / 'evidence/source_hashes.json', {name: sha(ROOT / name) for name in sources})
    with (OUT / 'evidence/prediction_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['comparison', 'version', 'events', 'hits', 'recall_percent', 'precision_percent',
                         'false_or_late_per_fleet_hour', 'AP_percent'])
        for key, row in comparison['alerts'].items():
            writer.writerow(['Final V1 fixed 42 flights', key, row['events'], row['hits'], row['recall'] * 100,
                             row['precision'] * 100, row['false_or_late_per_fleet_h'], row['average_precision'] * 100])
    return metrics


def embedded_document(path):
    text = path.read_text(encoding='utf-8')
    def inline_image(match):
        image = path.parent / match[1]
        return 'src="data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii') + '"'
    text = re.sub(r'src="([^"\s]+\.png)"', inline_image, text)
    for filename in ('history-data.json', '30초_이력확장_분석.md'):
        source = path.parent / filename
        if filename == 'history-data.json' and not source.exists():
            source = path.parent / 'long_history.json'
        if source.exists():
            mime = 'application/json' if source.suffix == '.json' else 'text/markdown'
            uri = 'data:' + mime + ';base64,' + base64.b64encode(source.read_bytes()).decode('ascii')
            text = text.replace('href="' + filename + '"', 'download="' + filename + '" href="' + uri + '"')
    bridge = '''<script>
document.addEventListener('click', function(event) {
  const anchor = event.target.closest('a[href]');
  if (!anchor) return;
  const destination = anchor.getAttribute('href');
  const panels = {'./':'replay','history.html':'history','fast-control.html':'fast','precursor.html':'precursor'};
  if (panels[destination]) {
    event.preventDefault();
    parent.postMessage({type:'capstone-section',panel:panels[destination]}, '*');
  } else if (/^https?:/.test(destination)) {
    anchor.target = '_blank';
    anchor.rel = 'noopener noreferrer';
  }
});
</script>'''
    return text.replace('</body>', bridge + '</body>')


def build_html():
    metrics = collect_metrics()
    template = (ROOT / 'tmp/team_report_template_20261107.html').read_text(encoding='utf-8')
    template = template.replace('@@METRICS@@', json.dumps(metrics, ensure_ascii=False).replace('</', '<\\/'))
    documents = dict(replay=ROOT / 'final_v1/Final V1 재배치 재생.html',
                     history=ROOT / 'output/outage_history_20261105/history.html',
                     precursor=ROOT / 'output/precursor_drops_20261105/precursor.html',
                     fast=ROOT / 'output/final_v1_fast_summary_20261105/fast-control.html')
    for key, path in documents.items():
        payload = base64.b64encode(embedded_document(path).encode('utf-8')).decode('ascii')
        template = template.replace('@@' + key.upper() + '@@', payload)
    for key, filename in (('DISTRIBUTION', '01_pattern_distribution.png'),
                          ('CURVES', '02_actual_pattern_curves.png'),
                          ('CAUSES', '03_causes_and_thresholds.png')):
        image = ROOT / 'output/outage_patterns_20261105' / filename
        uri = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')
        template = template.replace('@@' + key + '@@', uri)
    if re.search(r'@@[A-Z_]+@@', template):
        raise ValueError('Unfilled report template')
    (OUT / REPORT_NAME).write_text(template, encoding='utf-8')
    print('Standalone HTML built', (OUT / REPORT_NAME).stat().st_size, flush=True)


def finalize():
    files = [path for path in walk_files(OUT, {'__pycache__', 'cache', 'numba_cache', '.venv'})
             if path.name not in ('FILE_MANIFEST.json', 'DELIVERY.json') and path.suffix not in ('.lock', '.log')]
    manifest = {path.relative_to(OUT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files}
    write_json(OUT / 'FILE_MANIFEST.json', dict(version='Final V1 team handoff', files=manifest,
               source_packages=PACKAGES, python_sources=sum(name.endswith('.py') for name in manifest),
               excludes=['Python environments', 'regenerable caches', 'earlier raw training tensors',
                         'Git credentials and site deployment source'],
               original_sources_preserved=True))
    archive = OUT.parent / 'Final_V1_팀공유_전체코드_HTML_20261107.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in files + [OUT / 'FILE_MANIFEST.json']:
            zipped.write(path, 'Final_V1_Team/' + path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        bad_file = zipped.testzip()
        if bad_file:
            raise ValueError('ZIP CRC failure: ' + bad_file)
    delivery = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                    files=len(files) + 1, report=str(OUT / REPORT_NAME), zip_crc_verified=True)
    write_json(OUT / 'DELIVERY.json', delivery)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('code', 'html', 'zip'))
    phase = parser.parse_args().phase
    OUT.mkdir(parents=True, exist_ok=True)
    {'code': build_code, 'html': build_html, 'zip': finalize}[phase]()
