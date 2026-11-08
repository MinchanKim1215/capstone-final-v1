"""Short training-map fixture for actual fresh capture, inputs and baseline."""
from . import ARTIFACTS,PARENT
from .contracts import read,write,now
from .confirmation import collect_job,prepare_case,parent_capture


def main():
    from adaptive_joint_expanded import ARTIFACTS as OLD
    r=read(PARENT/'data_manifest.json')['records'][0]['source_record'];folder=ARTIFACTS/'fixture';folder.mkdir(exist_ok=True)
    parent_capture.initialize();task=(r['spec'],read(OLD/'protocol.json')['config'],r['deployment'],True,12,str(folder/'collector_train12'))
    record=collect_job(task);item=prepare_case(record)
    assert item['queries']>0 and item['reference_path']
    v=dict(passed=True,created_at=now(),training_region_only=True,not_in_training_or_evaluation=True,steps=12,queries=item['queries'],key=record['key'],
        actual_capture_inputs_and_frozen_baseline=True)
    write(ARTIFACTS/'collector_preflight.json',v);print(v,flush=True)


if __name__=='__main__':main()
