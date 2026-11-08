"""기존 48개 점수 캐시에 새 알림 정책 적용. 신경망 추론·학습 변경 없음."""
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json
from usable.data import collection
from usable.evaluate import RESULT,RUNS,calibrate
from usable.alarm_policy import arrays_for_record,simulate_policy,summarize


def main():
    folder=RUNS/(read_json(RESULT/'selection.json')['selected_variant']+'_s11');policy=read_json(folder/'alarm_policy.json');cal=calibrate(folder)
    manifest,_=collection();new=np.zeros(9,np.int64);old=np.zeros(9,np.int64);duration=0
    for r in [r for r in manifest['records'] if r['split']=='test']:
        with np.load(folder/'runtime'/(r['id']+'.npz')) as z:index,p=z['index'],z['probability']
        a=arrays_for_record(r,index,p);c=policy['selected']
        if c:new+=simulate_policy(*a,c['enter_threshold'],c['exit_threshold'])[0]
        old+=simulate_policy(*a,cal['threshold_conservative'],cal['threshold_conservative'])[0]
        duration+=r['metadata']['duration_s'];print('alarm policy applied',r['id'],flush=True)
    out={'new_policy':summarize(new,duration),'previous_point_warning':summarize(old,duration),'fit_uses_these_records':False,
         'status':'exploratory reuse of previously evaluated flights; separate new-random confirmation provided'}
    write_json(RESULT/'alarm_policy_existing_test.json',out)


if __name__=='__main__':main()
