"""Default research entry point for the operating option selected by the user."""
from .profile import load_profile
from forecast_recall.runtime_options import load_service as load_verified_service


def load_service(city,n_drones,n_search,nominal=None,device='cpu'):
    profile=load_profile()
    stream=load_verified_service(city,n_drones,n_search,nominal,device,
        budget_per_hour=profile['calibration_FA_budget_per_hour'])
    if stream.policy['threshold']!=profile['threshold']:
        raise ValueError('Loaded stream differs from the selected profile')
    return stream
