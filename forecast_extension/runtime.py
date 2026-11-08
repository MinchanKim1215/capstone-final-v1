"""Hash-bound research loader for the frozen V9 model and operating point."""
from pathlib import Path
from . import ARTIFACTS
from .contracts import read,sha
from .evaluate import verify_freeze,model_pin
from forecast_rebuild.runtime_stable import StableLocalForecast
from forecast_rebuild import ROOT as PARENT_ROOT


def runtime_dependencies():
    # This numerical adapter was added after the parent's model freeze, so
    # explicitly bind it as well as the parent sources verified by that freeze.
    pinned={str(PARENT_ROOT/'runtime_stable.py'):'ca216d3fd4ef3f595a5a0dc8e3d60ac3e9367fd57cefaaff626ebac80b06cd3a'}
    for path,expected in pinned.items():
        if sha(path)!=expected:raise ValueError('The audited numerical runtime adapter changed')
    return pinned


class Forecast(StableLocalForecast):
    def __init__(self,*args,operating_point='advisory',**kwargs):
        super().__init__(*args,**kwargs)
        if operating_point not in ('advisory','common_FA_budget1','precision90_FA1_partial'):raise ValueError('Unknown operating point')
        if operating_point!='advisory':
            choice=self.policy.get(operating_point)
            if not choice:raise ValueError('This operating point did not exist on calibration data')
            self.policy=dict(self.policy,threshold=choice['threshold'],enabled=False,mode=operating_point)
        self.operating_point=operating_point


def load_candidate(city,observer,n_drones,n_search,nominal=None,device='cpu',operating_point='advisory'):
    frozen=verify_freeze();candidate=read(ARTIFACTS/'research_candidate.json')
    if candidate['model']!=frozen['selected_model']:raise ValueError('Candidate differs from the pre-confirmation choice')
    if sha(Path(__file__))!=candidate['runtime_sha256'] or sha(ARTIFACTS/'freeze.json')!=candidate['freeze_sha256']:
        raise ValueError('Research implementation differs from its audited bundle')
    if candidate['runtime_dependencies']!=runtime_dependencies():raise ValueError('Research runtime dependency pins differ')
    if model_pin(candidate['model'])!=candidate['model_pin']:raise ValueError('Candidate weights differ')
    if sha(candidate['policy'])!=candidate['policy_sha256']:raise ValueError('Candidate alarm policy differs')
    return Forecast(candidate['model'],city,observer,n_drones,n_search,policy=candidate['policy'],nominal=nominal,device=device,
                    operating_point=operating_point,confirmation_goal_met=False,hardware_validated=False)

