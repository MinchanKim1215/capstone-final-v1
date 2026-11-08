"""Commit experimental choices after diagnosis and before fitting models."""
from . import ARTIFACTS
from .contracts import read,write,sha,now,prepare,SEED


def main():
    prepare();diagnosis=ARTIFACTS/'diagnostics/results.json'
    if not diagnosis.exists():raise ValueError('Complete and review stage1 before defining training')
    value=dict(created_at=now(),diagnosis_sha256=sha(diagnosis),seeds=[SEED,SEED+10],
        physical_task=dict(truth='Unchanged >=1s physical-link onset and [3,7) first-alert rules',
            training='Same immutable V9 merged1123200 rows',variants=['uniform','targeted_weights'],
            references=['Frozen V9 tree','Frozen V9 selected blend75']),
        service_task=dict(truth='Global search-fleet packet deadline service degradation, predicted from one relay local view',
            deadline_s=.25,rolling_window_s=1.,minimum_timely_pdr=.9,minimum_settled_packets=20,persistent_s=1.,
            valid_births='At q*.5 include births in (q*.5-.25-1, q*.5-.25], so every counted deadline has elapsed',
            primary_observer='First relay (physical index n_search), fixed before training',
            training='V9 fresh245 physical training flights;84 augmentation variants share identical service labels and disjoint query quotas',
            selection='V9 fresh45 selection flights',calibration='V9 fresh45 calibration flights',
            variants=['uniform_local','targeted_local','targeted_local_search_pair_geometry'],
            reference='Maximum supported raw V9 blend link risk of the same observer, calibrated separately to service truth',
            caution='A new prediction task; AP and recall cannot be directly subtracted from physical-link-task results. Packet deadline and PDR criteria are study assumptions, not hardware requirements.'),
        targeted_weights=dict(positive_target_fraction=.15,maximum_positive_multiplier=8.,hard_negative_multiplier=2.,
            mining='Train rows only, from corresponding uniform seed-ensemble scores, negatives above its training-negative90th percentile, no validation/confirmation mining'),
        topology_inputs='Current and extrapolated search-search geometry using only this relay received positions, velocities, issued targets, timestamps and optional received energy. Public GIS only; no true future position/action/jammer/packet outcome.',
        tree=dict(max_rounds=800,early_stopping=60,num_leaves=31,learning_rate=.04,min_data_in_leaf=100,lambda_l2=5.),
        candidate_definition='Each variant is the equal mean of both predeclared seeds; do not choose the better seed',
        selection='For each task: maximize recall among selection precision>=.9, false_or_late/h<=1,min50 alerts; otherwise mean AP5/AP37',
        calibration='Platt maps and operating points from calibration45 only; preserve raw.5/advisory/FA1/P90-partial',
        confirmation='42 new fixed-v7-policy flights on known14 confirmation maps with new independent seeds, generated after model/policy freeze',
        original_final_test_opened=False,automatic_PPO_replacement=False,production_ready=False)
    path=ARTIFACTS/'experiment_plan.json'
    if path.exists():
        old=read(path)
        for k in value:
            if k!='created_at' and old[k]!=value[k]:raise ValueError('Committed experiment plan changed')
    else:write(path,value)
    print(path)


if __name__=='__main__':main()
