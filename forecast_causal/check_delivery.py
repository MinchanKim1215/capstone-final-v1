"""Verify all result rows and visually inspected figures before completion."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,milestone
from .evaluate import verify_freeze
from .report import NAMES,CAUSES

def main():
    verify_freeze();r=read(ARTIFACTS/'final_results.json');text=(ROOT/'RESULTS_20261103.md').read_text(encoding='utf-8');checks=[]
    rows=[[v.strip() for v in line.split('|')[1:-1]] for line in text.splitlines() if line.startswith('|')]
    percent=lambda v:'-' if v is None else f'{100*v:.2f}%'
    for tag,m in r['models'].items():
        a=m['calibrated']['alarms'];label=NAMES[tag]+(' **사전 선정**' if tag==r['selected_tag'] else '')
        expected=[label,percent(a['event_recall_3s']),str(a['event_hits_3s'])+'/'+str(a['eligible_directed_link_events']),percent(a['precision']),str(a['timely_correct_first_alerts'])+'/'+str(a['first_alerts']),f'{a["false_or_late_per_drone_hour"]:.2f}']
        ranking=[NAMES[tag],percent(m['raw']['next5_on_common_currently_up_links']['ap']),percent(m['raw']['any3to7_on_common_support']['ap']),percent(m['raw']['any3to7_on_common_support']['auroc']),f'{m["calibrated"]["calibrated_any3to7"]["brier"]:.5f}']
        if expected not in rows or ranking not in rows:raise ValueError('Report numbers mismatch: '+tag)
        checks.append(tag)
    causes=r['cause_analysis'];selected=r['selected_tag']
    for key,label in CAUSES.items():
        n=causes['counts'].get(key,0);expected=[label,str(n),f'{causes["hits"]["frozen_v13"].get(key,0)}/{n}',f'{causes["hits"][selected].get(key,0)}/{n}']
        if expected not in rows:raise ValueError('Cause report row mismatch: '+key)
    if sha(ARTIFACTS/'final_results.json') not in text:raise ValueError('Report source binding absent')
    for name in ('runtime_checks.json','audit.json','collector_preflight.json','input_preflight.json','model_preflight.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required verification failed')
    prov=read(ARTIFACTS/'figures/provenance.json')
    if prov['final_results_sha256']!=sha(ARTIFACTS/'final_results.json') or prov['selection_causes_sha256']!=sha(ARTIFACTS/'selection_event_causes.json'):raise ValueError('Figure source mismatch')
    qa=read(ARTIFACTS/'figures/visual_qa.json')
    figures={p.name:sha(p) for p in (ARTIFACTS/'figures').glob('*.png')}
    if not qa['all_four_figures_viewed_and_passed'] or qa['figures']!=figures:raise ValueError('Actual final plot viewing required')
    write(ARTIFACTS/'delivery_checks.json',dict(passed=True,created_at=now(),alarm_rows_verified=checks,ranking_rows_verified=len(checks),report_sha256=sha(ROOT/'RESULTS_20261103.md'),final_results_sha256=sha(ARTIFACTS/'final_results.json'),figures=figures,original_final_test_opened=False))
    milestone(4,'RESULTS_20261103.md·final_results.json·runtime_checks.json·audit.json·delivery_checks.json; 새42비행·실제API·377비행 보존·도표 확인 완료','후속 우선순위: 물리·발행 경로 예측을 정교화하고 점수 미달123건·이미 켜진 경보13건·간섭 그룹5/29건을 분석한다. 새 큐 입력·GRU의 추가 효과는 확인되지 않았다. 현재 프로필은 보존하며 다음 실험도 새 확인 자료·동일 경보 비용으로 검증한다.')
    state=dict(state='completed',stage='all_four_stages_training_new_confirmation_runtime_audit_report_completed',updated_at=now(),production_ready=False,hardware_validated=False,original_final_test_opened=False,automatic_PPO_replacement=False)
    write(ARTIFACTS/'status.json',state)
    from .ledger import refresh
    refresh(state)

if __name__=='__main__':main()
