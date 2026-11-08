"""Finish the predeclared selection-only navigation-quality gate."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import traceback
import numpy as np
from . import ARTIFACTS
from .probe import evaluate
from forecast_graph.contracts import read,write,now,source_records


def main():
    try:
        records=[r for r in source_records() if r['role']=='selection'];results={}
        write(ARTIFACTS/'status.json',dict(state='running',stage='selection_navigation_quality_gate',complete=0,total=len(records),updated_at=now()))
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
            jobs={pool.submit(evaluate,r):i for i,r in enumerate(records)}
            for f in as_completed(jobs):
                results[jobs[f]]=f.result();write(ARTIFACTS/'status.json',dict(state='running',stage='selection_navigation_quality_gate',complete=len(results),total=len(records),updated_at=now()))
                print(len(results),'/',len(records),flush=True)
        original=float(np.sqrt(np.mean([np.mean(np.asarray(r['rmse_by_horizon']['original'][3:8])**2) for r in results.values()])))
        proposed=float(np.sqrt(np.mean([np.mean(np.asarray(r['rmse_by_horizon']['mission'][3:8])**2) for r in results.values()])))
        gain=1-proposed/original;passed=gain>=.30
        value=dict(created_at=now(),scope='Selection45 only',flights=len(records),weighting='Equal flight and horizon3..7 mean squared radial position error',
            original_rmse_m=original,mission_preview_rmse_m=proposed,relative_rmse_reduction=gain,required_reduction=.30,passed=passed,
            confirmation_used=False,no_RF_future_input=True,physical_trajectories_unchanged=True)
        write(ARTIFACTS/'gate_result.json',value)
        write(ARTIFACTS/'status.json',dict(state='gate_passed_pending_fit' if passed else 'completed',stage='navigation_quality_gate_complete',updated_at=now(),result=value))
        print(value,flush=True)
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='navigation_quality_gate',error=repr(error),updated_at=now()));traceback.print_exc();raise


if __name__=='__main__':main()

