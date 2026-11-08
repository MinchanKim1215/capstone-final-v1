"""相同 경로/잠재 채널에서 비콘 슬롯 예약 유무의 앱 패킷 비용 비교.

수신 시뮬레이션의 난수는 같은 초기 시드입니다. 전송 시점 변경으로 난수
소모 순서는 달라지므로 패킷별 동일 성공/실패를 강제로 공유하지 않습니다.
저장된 float32 채널을 두 조건에 동일하게 사용합니다.
"""
import numpy as np
from swarm_sim.regions import ROOT,write_json
from swarm_sim.fast_network import packet_run as without_beacons
from swarm_sim.config import Scenario
from upgrade.collect import prepare_city_without_unused_radio_cache
from usable.packet import packet_run
from usable.data import collection


def main():
    manifest,p=collection();rows=[]
    # 선정 성능과 무관하게 미리 정한 세 지역, 18개 전체 비행.
    keys=('teheran','migeum','seoul_20261023_03')
    for key in keys:
        e=next(x for x in p['maps'] if x['key']==key)
        c=prepare_city_without_unused_radio_cache(ROOT/e['path'],Scenario(),ROOT/'cache/usable')
        for r in manifest['records']:
            if r['region']!=key:continue
            with np.load(ROOT/r['path']/'truth.npz') as z:prob=z['hop_success_probability'].astype(np.float64)
            with np.load(ROOT/'cache/usable'/f"{key}_{r['metadata']['direction']}_motion.npz") as z:
                nodes=np.repeat(z['nodes'][:-1],6,axis=0)
            a=without_beacons(prob,nodes,c.visible,3,.5,r['metadata']['seed']+1000003)
            b=packet_run(prob,nodes,c.visible,3,.5,r['metadata']['seed']+1000003)
            def metrics(v):
                return {'generated':int(v[0][0]),'delivered':int(v[0][1]),'pdr':float(v[0][1]/v[0][0]),
                        'mean_delay_s':float(np.mean(v[4][v[3]==1])),'timely_delivery_fraction':float(v[0][8]/v[0][0])}
            rows.append({'id':r['id'],'without_beacon_slots':metrics(a),'with_beacon_slots':metrics(b)})
    out={'episodes':rows,'reserved_slot_fraction':.04,'beacon_rf_airtime_fraction':8*128*8/1e6,
         'rf_airtime_note':'1.024ms frame serialization within reserved 5ms slots; unmodelled synchronization/guard cost in remaining slot',
         'average_pdr_difference':float(np.mean([x['with_beacon_slots']['pdr']-x['without_beacon_slots']['pdr'] for x in rows])),
         'mean_delay_difference_s':float(np.mean([x['with_beacon_slots']['mean_delay_s']-x['without_beacon_slots']['mean_delay_s'] for x in rows]))}
    write_json(ROOT/'results/usable/beacon_cost.json',out)


if __name__=='__main__':main()
