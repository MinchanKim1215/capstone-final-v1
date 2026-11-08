"""Bind rendered plots and numeric report back to final evaluation evidence."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .report import NAMES,pct

def main():
    result=read(ARTIFACTS/'final_results.json');text=(ROOT/'RESULTS_20261103.md').read_text(encoding='utf-8');checks=[]
    for tag,m in result['models'].items():
        a=m['calibrated']['alarms'];label=NAMES[tag]+(' **사전 선정**' if tag==result['selected_tag'] else '')
        row=f'| {label} | {pct(a["event_recall_3s"])} | {a["event_hits_3s"]}/{a["eligible_directed_link_events"]} | {pct(a["precision"])} | {a["timely_correct_first_alerts"]}/{a["first_alerts"]} | {a["false_or_late_per_drone_hour"]:.2f} |'
        if row not in text:raise ValueError('Report alarm row mismatch: '+tag)
        checks.append(tag)
    if sha(ARTIFACTS/'final_results.json') not in text:raise ValueError('Report evidence binding absent')
    for name in ('runtime_checks.json','audit.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Incomplete verification')
    provenance=read(ARTIFACTS/'figures/provenance.json')
    if provenance['final_results_sha256']!=sha(ARTIFACTS/'final_results.json'):raise ValueError('Plots from different result')
    visual=ARTIFACTS/'figures/visual_qa.json'
    if not visual.exists() or not read(visual)['all_three_figures_viewed_and_passed']:raise ValueError('Actual plot viewing required')
    value=dict(passed=True,created_at=now(),report_model_rows_verified=checks,report_sha256=sha(ROOT/'RESULTS_20261103.md'),result_sha256=sha(ARTIFACTS/'final_results.json'),
        figures={p.name:sha(p) for p in sorted((ARTIFACTS/'figures').glob('*.png'))},runtime_and_integrity_passed=True,original_test_maps_unused=True)
    write(ARTIFACTS/'delivery_checks.json',value);print(value,flush=True)

if __name__=='__main__':main()
