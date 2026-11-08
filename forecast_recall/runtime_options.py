"""Research-only pre-result calibration operating points; default stays frozen.

This adapter changes no learned weights, probability or event definition. It
exposes the FA3/5 points already locked before confirmation prediction outcomes.
"""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .runtime import Stream


def export():
    plan=read(ARTIFACTS/'secondary_operating_points_plan.json');primary=read(ARTIFACTS/'research_candidate.json');tag=primary['candidate']['tag']
    if plan['primary_freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):raise ValueError('Operating points belong to different study')
    value=dict(created_at=now(),primary_export_sha256=sha(ARTIFACTS/'research_candidate.json'),
        plan_sha256=sha(ARTIFACTS/'secondary_operating_points_plan.json'),adapter_sha256=sha(__file__),selected_tag=tag,
        available_budgets=[int(b) for b,p in plan['items'][tag]['points'].items() if p is not None],
        default='Original primary operating point remains unchanged',production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,
        timing_note='Adapter added after results; all exposed thresholds were fixed from calibration in the pre-result secondary plan, not selected on confirmation.')
    path=ARTIFACTS/'research_operating_options.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Operating adapter export changed')
    else:write(path,value)
    return value


class BudgetStream(Stream):
    def __init__(self,city,n,ns,nominal=None,device='cpu',budget_per_hour=None):
        super().__init__(city,n,ns,nominal,device);self.budget_per_hour=budget_per_hour
        if budget_per_hour is None:return
        b=read(ARTIFACTS/'research_operating_options.json')
        if sha(__file__)!=b['adapter_sha256'] or sha(ARTIFACTS/'research_candidate.json')!=b['primary_export_sha256'] or sha(ARTIFACTS/'secondary_operating_points_plan.json')!=b['plan_sha256']:raise ValueError('Operating adapter changed')
        if budget_per_hour not in b['available_budgets']:raise ValueError('No calibration point with>=50alerts for supplied budget')
        plan=read(ARTIFACTS/'secondary_operating_points_plan.json');entry=plan['items'][b['selected_tag']]
        if sha(entry['policy'])!=entry['policy_sha256']:raise ValueError('Probability calibration changed')
        self.policy=dict(self.policy,threshold=entry['points'][str(budget_per_hour)]['threshold']);self.active=False

    def observe(self,*args,**kwargs):
        out=super().observe(*args,**kwargs);out['calibration_FA_budget_per_hour']=self.budget_per_hour;return out


def load_service(city,n_drones,n_search,nominal=None,device='cpu',budget_per_hour=None):
    return BudgetStream(city,n_drones,n_search,nominal,device,budget_per_hour)


if __name__=='__main__':export()
