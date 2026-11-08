"""TRAIN-only short real collection, capture and cohort/reference construction."""
from . import ARTIFACTS
from .contracts import source_records,read,write,now
from .confirmation import initialize,collect_job,prepare_case


def main():
    from adaptive_joint_expanded import ARTIFACTS as OLD
    source=source_records()[0];folder=ARTIFACTS/'fixture';folder.mkdir(exist_ok=True)
    initialize();task=(source['spec'],read(OLD/'protocol.json')['config'],source['deployment'],True,12,str(folder/'train12'))
    record=collect_job(task);record['role']='train';item=prepare_case(record)
    from .episodes import cohort
    a=cohort(item,'link');b=cohort(item,'service')
    if not len(a.ends) or len(a.ends)!=len(b.ends):raise ValueError('Empty or unmatched fixture cohorts')
    value=dict(passed=True,created_at=now(),key=record['key'],steps=12,queries=item['queries'],region=record['region'],
        training_region_only=True,not_in_training_or_evaluation=True,extended_payload_bytes=128,
        post_flight_capture_and_parent_reference=True)
    write(ARTIFACTS/'collector_preflight.json',value);print(value,flush=True)


if __name__=='__main__':main()
