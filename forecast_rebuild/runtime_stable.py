"""Numerically align online public-plan arithmetic with the frozen batch path.

The offline planning contract explicitly computes telemetry in float64 before
writing float32 features. One float32 ULP can cross a tree split. This adapter
applies exactly that same conversion online; it changes no model, target,
sensor entitlement, feature meaning or confirmation calculation. The frozen
base implementation remains intact for audit.
"""
import numpy as np
from .runtime import LocalForecast
from .online_planning import PlanContext


class StablePlanContext(PlanContext):
    def compute(self,x,m,c,clock_s):
        return super().compute(np.asarray(x,np.float64),m,np.asarray(c,np.float64),clock_s)


class StableLocalForecast(LocalForecast):
    def reset(self):
        super().reset()
        if self.plan is not None:
            self.plan=StablePlanContext(self.city,self.nominal,self.observer,self.ns)


def load_research_candidate(city,observer,n_drones,n_search,nominal=None,device='cpu'):
    """Load the exact audited candidate; operational alarms remain disabled."""
    from pathlib import Path
    from . import ARTIFACTS
    from .contracts import read,sha
    from .calibration import verify_freeze
    frozen=verify_freeze();candidate=read(ARTIFACTS/'research_candidate.json')
    if candidate['model']!=frozen['selected_model'] or candidate['alarm_policy']!=frozen['selected_alarm']:
        raise ValueError('Research bundle differs from the frozen comparison')
    for path,expected in [(candidate['model'],candidate['model_sha256']),
                          (candidate['alarm_policy'],candidate['alarm_policy_sha256']),
                          (candidate['freeze'],candidate['freeze_sha256']),
                          (Path(__file__),candidate['runtime_adapter_sha256'])]:
        if sha(path)!=expected:raise ValueError('Research bundle integrity check failed')
    return StableLocalForecast(candidate['model'],city,observer,n_drones,n_search,
        policy=candidate['alarm_policy'],nominal=nominal,device=device,
        confirmation_goal_met=False,hardware_validated=False)
