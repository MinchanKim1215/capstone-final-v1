"""선택 동결 → 보정 → 공통 표본/전체 비행/실행 경고의 독립 평가."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from swarm_sim.regions import read_json,write_json
from usable.evaluate import RESULT,RUNS,calibrate,banks,dense,freeze_final_selection
from usable.runtime_evaluate import main as runtime
from upgrade.train import configure


def main():
    sel=freeze_final_selection();selected=sel['selected_variant'];configure(11,2)
    names=[k+'_s11' for k in sel['variants']]+[selected+'_s22',selected+'_s33']
    for name in names:
        c=calibrate(RUNS/name);print('calibrated',name,c['threshold_balanced'],c['threshold_conservative'],flush=True)
    # PyTorch 스레드 초기화 이후의 fork를 피하도록 spawn을 사용합니다.
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(dense,'B0_legacy_inputs_s11'),pool.submit(dense,selected+'_s11'),pool.submit(runtime)]
        banks()
        for future in as_completed(jobs):
            r=future.result();print('evaluation completed',None if r is None else r.get('run'),flush=True)
    write_json(RESULT/'evaluation_complete.json',{'complete':True,'selected_run':selected+'_s11','test_used_for_selection':False,
                                                'test_geography_previously_inspected':True})


if __name__=='__main__':main()
