"""평가 지역의 정적 지도 파생 입력. 모형 선택에 평가 정답 미사용."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from swarm_sim.regions import ROOT,read_json,write_json
from usable.data import collection
from usable.spatial_data import one_region


def main():
    selection=read_json(ROOT/'results/usable/selection.json')
    if 'variants' not in selection:raise ValueError('지도 보강 변형의 검증 선택 기록 필요')
    manifest,p=collection();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(one_region,e,[r for r in manifest['records'] if r['region']==e['key']]) for e in p['maps'] if e['split']=='test']
        for f in as_completed(jobs):rows+=f.result()
    write_json(ROOT/'results/usable/test_spatial_input_audit.json',{'files':rows,'selection_at_generation':selection,
                                                                 'test_labels_used_for_selection':False})


if __name__=='__main__':main()
