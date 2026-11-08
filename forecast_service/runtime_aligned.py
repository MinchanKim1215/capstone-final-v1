"""Align runtime contracts with the unchanged profile and >=21s alarm metric.

The frozen forecasters still emit their first probabilities at20s. Suppress
only warm-up alarm transitions and reset their latch before the21s evaluation
start. No input, probability, model, calibration or benchmark result changes.
"""
from . import ARTIFACTS
from .contracts import read,sha
from . import runtime as frozen_runtime
import numpy as np


def check_adapter():
    expected=read(ARTIFACTS/'research_candidates.json')['alarm_start_adapter_sha256']
    if sha(__file__)!=expected:raise ValueError('Audited alarm-start adapter changed')


def validate_service_profile(city,n_drones,n_search):
    if (n_drones,n_search) not in ((6,4),(8,3),(10,6)):raise ValueError('Use one of the study public fleet configurations')
    scenario=getattr(city,'_forecast_scenario',None)
    profile=dict(frequency_hz=2.4e9,altitude_m=50.,tx_power_dbm=10.,gain_dbi=0.,bandwidth_hz=1e6,noise_figure_db=7.,
        bit_rate_bps=1e6,link_success_threshold=.9,shadow_sigma_db=1.5,shadow_correlation_s=4.,slot_s=.005)
    if scenario is None or any(not np.isclose(getattr(scenario,k,float('nan')),v) for k,v in profile.items()) or scenario.mac!='tdma' or scenario.radio_model!='bullington':
        raise ValueError('The service model has not been validated for this radio/flight profile')


class AlignedStream:
    def __init__(self,stream,task):self.stream=stream;self.task=task

    def __getattr__(self,name):return getattr(self.stream,name)

    def observe(self,time_end_s,*args,**kwargs):
        value=self.stream.observe(time_end_s,*args,**kwargs)
        if time_end_s<21:
            if self.task=='link':
                self.stream.active[:]=False;self.stream.qualified[:]=False
                for row in value.get('links',[]):row.update(advisory_first_alert=False,qualified_first_alert=False,trusted_alarm=False)
            else:
                self.stream.active=False;value.update(first_alert=False,trusted_alarm=False)
        value['alarm_start_s']=21.;return value


def load_link(*args,**kwargs):check_adapter();return AlignedStream(frozen_runtime.load_link(*args,**kwargs),'link')


def load_service(*args,**kwargs):
    check_adapter();city=kwargs.get('city',args[0] if len(args)>0 else None)
    n=kwargs.get('n_drones',args[2] if len(args)>2 else None);ns=kwargs.get('n_search',args[3] if len(args)>3 else None)
    validate_service_profile(city,n,ns);return AlignedStream(frozen_runtime.load_service(*args,**kwargs),'service')
