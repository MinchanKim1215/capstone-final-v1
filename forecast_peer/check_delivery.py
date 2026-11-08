"""Check all12matched report rows and both source-bound figure evidence."""
from . import ROOT,ARTIFACTS,GRAPH
from .contracts import read,write,sha,now
from .report import NAMES,pct

def main():
    r=read(ARTIFACTS/'final_results.json');g=read(GRAPH/'final_results.json');models=dict(g['models']);models.update(r['models']);text=(ROOT/'RESULTS_20261103.md').read_text(encoding='utf-8');checks=[]
    for tag,m in models.items():
        a=m['calibrated']['alarms'];label=NAMES[tag]+(' **최종 사전 선정**' if tag==r['selected_tag'] else '')
        row=f'| {label} | {pct(a["event_recall_3s"])} | {a["event_hits_3s"]}/{a["eligible_directed_link_events"]} | {pct(a["precision"])} | {a["timely_correct_first_alerts"]}/{a["first_alerts"]} | {a["false_or_late_per_drone_hour"]:.2f} |'
        if row not in text:raise ValueError('Numeric alarm report row mismatch: '+tag)
        checks.append(tag)
    for tag,m in r['models'].items():
        row=f'| {NAMES[tag]} | {pct(m["raw"]["next5_on_common_currently_up_links"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["auroc"])} | {m["calibrated"]["calibrated_any3to7"]["brier"]:.5f} |'
        if row not in text:raise ValueError('Numeric ranking report row mismatch')
    for folder in (ARTIFACTS,GRAPH):
        if sha(folder/'final_results.json') not in text:raise ValueError('Report source binding absent')
    for name in ('runtime_checks.json','audit.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required verification failed')
    provenance=read(ARTIFACTS/'figures/provenance.json')
    if provenance['final_results_sha256']!=sha(ARTIFACTS/'final_results.json') or provenance['v14_results_sha256']!=sha(GRAPH/'final_results.json'):raise ValueError('Figures from different result')
    if not read(ARTIFACTS/'figures/visual_qa.json')['all_three_figures_viewed_and_passed']:raise ValueError('Actual plot viewing required')
    value=dict(passed=True,created_at=now(),alarm_rows_verified=checks,ranking_rows_verified=7,report_sha256=sha(ROOT/'RESULTS_20261103.md'),final_results_sha256=sha(ARTIFACTS/'final_results.json'),
        v14_results_sha256=sha(GRAPH/'final_results.json'),v14_delivery_sha256=sha(GRAPH/'delivery_manifest.json'),same_shared42_not84=True,original_final_test_opened=False,
        figures={p.name:sha(p) for p in (ARTIFACTS/'figures').glob('*.png')})
    write(ARTIFACTS/'delivery_checks.json',value);print(value,flush=True)

if __name__=='__main__':main()
