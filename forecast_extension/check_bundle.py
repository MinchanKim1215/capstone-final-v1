"""Exercise the exported, hash-verified loader on actual causal input frames."""
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now
from .runtime import load_candidate
from .evaluate import verify_freeze
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context


def main():
    frozen=verify_freeze();candidate=read(ARTIFACTS/'research_candidate.json')
    runtime=read(ARTIFACTS/'runtime_checks.json');corpus=read(ARTIFACTS/'dataset.json')
    record=next(r for r in corpus['rows'] if r['key']==runtime['tested_record_key'])
    episode=Episode(record);city,nominal=public_context(record);ns=record['n_search'];n=episode.values.shape[2]
    policy=read(candidate['policy']);checks=[];reference=None;maximum=0.
    for point in ('advisory','common_FA_budget1','precision90_FA1_partial'):
        if point!='advisory' and not policy.get(point):continue
        predictor=load_candidate(city,ns,n,ns,nominal=nominal,device='cpu',operating_point=point)
        expected=policy['threshold'] if point=='advisory' else policy[point]['threshold']
        assert predictor.policy['threshold']==expected
        risks=[];supported=0
        for q in range(80):
            value=predictor.observe((q+1)*.5,episode.values[q,0],episode.masks[q,0],episode.intents[q,0],episode.energy[q,0])
            for link in value.get('links',[]):
                assert not link['trusted_alarm']
                risks.append([np.nan if link[key] is None else link[key] for key in ('risk_5s','risk_3to7s')])
                supported+=int(link['risk_5s'] is not None)
        assert value['status']=='ready' and supported>0
        risks=np.asarray(risks,float)
        if reference is None:reference=risks
        else:
            np.testing.assert_allclose(risks,reference,atol=1e-7,rtol=1e-6,equal_nan=True)
            finite=np.isfinite(reference);maximum=max(maximum,float(np.max(np.abs(risks[finite]-reference[finite]))))
        checks.append(dict(operating_point=point,threshold=expected,input_frames=80,supported_predictions=supported,trusted_alarm_disabled=True))
    assert not candidate['production_ready'] and not candidate['hardware_validated'] and not candidate['automatic_PPO_replacement']
    write(ARTIFACTS/'bundle_checks.json',dict(passed=True,created_at=now(),checks=checks,
        selected_tag=candidate['selected_tag'],exported_bundle_sha256=sha(ARTIFACTS/'research_candidate.json'),
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),checker_sha256=sha(ROOT/'check_bundle.py'),
        operating_points_max_probability_difference=maximum,existing_PPO_replaced=False))
    print(ARTIFACTS/'bundle_checks.json')


if __name__=='__main__':main()
