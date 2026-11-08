"""Continue existing collection workers, then use final exact metric code."""
import time
from . import ARTIFACTS
from .contracts import read,write,now
from .campaign import stage,run,start,finish


def main():
    if not (ARTIFACTS/'stage1_comparison.json').exists():raise ValueError('Stage1 comparison must be committed before controller handoff')
    stage('stage2_fresh_experience_collection')
    for filename in ('collection_status.json','stream_preparation_status.json'):
        while True:
            path=ARTIFACTS/filename
            if path.exists():
                state=read(path)
                if state['state']=='failed':raise RuntimeError('Worker stage failed: '+filename)
                if state['state']=='completed':break
            time.sleep(5)
    stage('stage2_finalize_data_and_observation_augmentation')
    run('forecast_extension.data',[],'prepare_data')
    run('forecast_extension.build_samples',[],'new_samples')
    run('forecast_extension.build_samples',['--merged'],'merged_tree_samples')
    stage('stage2_new_experience_training')
    neural=start('forecast_extension.train',['--tag','new_data_20000','--new-samples','forecast_extension/artifacts/samples/new_only'],'new_data_training')
    run('forecast_extension.train_tree',['--tag','new_data_tree'],'new_tree_energy')
    run('forecast_extension.train_tree',['--tag','new_data_tree_no_energy','--no-energy'],'new_tree_no_energy')
    finish(neural,'new_data_training')
    run('forecast_extension.finish',[],'final_comparison')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',error=repr(error),updated_at=now(),original_final_test_opened=False));raise
