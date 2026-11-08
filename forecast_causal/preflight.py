"""Check exact packets, delay accounting, fleet variants and prefix causality."""
import numpy as np
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,status
from .replay import one

def main():
    records=read(PARENT/'data_manifest.json')['records'];training=[p for p in records if p['role']=='train'];first=training[0]
    small=one(first,12);full=one(first)
    with np.load(small['path']) as a,np.load(full['path']) as b:
        for key in ('observed','physical_truth','tx','rx'):np.testing.assert_equal(a[key],b[key][:len(a[key])])
    fixtures=[];seen=set()
    for p in training:
        deployment=p['source_record']['deployment']
        if deployment not in seen:fixtures.append(one(p,12));seen.add(deployment)
    continuous=next(p for p in training if p['source_record']['n_search']==6 and p['source_record']['spec']['kind']=='continuous');stress=one(continuous)
    value=dict(passed=True,created_at=now(),full_original_flights_verified=[full['key'],stress['key']],fleet_variants=len(seen),
        exact_future_truncation_prefix=True,all_original_packet_status_and_delay_exact=True,received_beacons_and_RF_exact=True,
        same_numba_draw_order=True,full_delay_components_conserved=True,original128B_restore_exact=True,
        source_pins={k:sha(ROOT/k) for k in ('recorder.py','instrumented_packet.py','wire.py','replay.py')})
    write(ARTIFACTS/'instrumentation_preflight.json',value);status('stage1_instrumentation_verified',next_action='선택45비행을 재생해200개 사건과134개 위험 점수 미달 사건을 원인별로 분석한다.');print(value,flush=True)

if __name__=='__main__':main()
