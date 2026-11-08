"""선택/확률 보정/경고 문턱 결정 후 평가. 기존에 살펴본 지역의 재검증."""
import argparse,math,json,time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import precision_recall_curve
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from swarm_sim.forecast_evaluation import binary_metrics
from upgrade.evaluate import grouped,threshold_at_fpr,event_rows,summarize_events
from upgrade.train import configure
from usable.data import Bank,PREPARED,collection,materialize,index_episode,load_observation
from usable.model import RiskTransformer
from usable.train import RUNS,VARIANTS,predict_bank

RESULT=ROOT/'results/usable'


def bank_for_state(name,state):
    if max(state['feature_indices'])>=48:
        from usable.spatial_train import SpatialBank
        return SpatialBank(name,state['feature_indices'])
    return Bank(name,state['feature_indices'])


def load(folder):
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    meta=read_json(folder/'run.json')
    if sha256_file(folder/'best.pt')!=meta['checkpoint_sha256']:raise ValueError('체크포인트 해시 불일치')
    model=RiskTransformer(**state['spec']);model.load_state_dict(state['state_dict']);model.eval()
    return model,state,meta


def freeze_selection():
    rows={k:read_json(RUNS/f'{k}_s11/run.json') for k in VARIANTS}
    selected=max(rows,key=lambda k:rows[k]['validation_select_pooled_ap'])
    out={'selected_variant':selected,'representative_seed':11,'replication_seeds':[22,33],
         'selection_metric':'highest validation-select pooled AP at 5s; before new test predictions',
         'validation_select_pooled_ap':{k:v['validation_select_pooled_ap'] for k,v in rows.items()},
         'checkpoint_sha256':{k:v['checkpoint_sha256'] for k,v in rows.items()},
         'test_previously_inspected_geography':True,'ensemble':False,
         'evaluation_plan':'all variants + selected replications on same 8192/region bank; representative selected and B0 on all available eligible test windows'}
    path=RESULT/'selection.json'
    if path.exists() and read_json(path)!=out:raise ValueError('선택 결과 동결 위반')
    write_json(path,out);return out


def freeze_final_selection():
    from usable.spatial_train import VARIANT
    from usable.spatial_single import VARIANT as SINGLE
    names=list(VARIANTS)+[VARIANT,SINGLE]
    rows={k:read_json(RUNS/f'{k}_s11/run.json') for k in names}
    selected=max(rows,key=lambda k:rows[k]['validation_select_pooled_ap'])
    out={'selected_variant':selected,'representative_seed':11,'replication_seeds':[22,33],
         'selection_metric':'highest validation-select pooled AP at 5s over five variants; before test predictions',
         'validation_select_pooled_ap':{k:v['validation_select_pooled_ap'] for k,v in rows.items()},
         'checkpoint_sha256':{k:v['checkpoint_sha256'] for k,v in rows.items()},
         'test_previously_inspected_geography':True,'ensemble':False,'variants':names,
         'additional_variant_reason':'validation-only spatial anticipation input refinement, separately frozen plan',
         'evaluation_plan':'five variants and selected replications on common 8192/region bank; selected and B0 on all eligible test windows'}
    path=RESULT/'selection.json';old=read_json(path)
    if old.get('variants')==names and old!=out:raise ValueError('최종 선택 동결 위반')
    if 'variants' not in old:write_json(RESULT/'initial_three_variant_selection.json',old)
    elif old.get('variants')!=names:write_json(RESULT/'four_variant_selection_before_auxiliary_control.json',old)
    write_json(path,out);return out


def calibrate(folder):
    path=folder/'calibration.json'
    if path.exists():
        old=read_json(path)
        if old['checkpoint_sha256']!=sha256_file(folder/'best.pt'):raise ValueError('보정값과 모형 가중치 불일치')
        return old
    model,state,_=load(folder);bank=bank_for_state('val_cal',state)
    logits=predict_bank(model,bank)[:,0].astype(float);y=bank.selected[:,4]
    # 1개 온도 계수만 적합합니다. 순위와 AP는 바뀌지 않습니다.
    def nll(logt):
        z=logits/np.exp(logt);return np.mean(np.logaddexp(0,z)-y*z)
    fit=minimize_scalar(nll,bounds=(-3,3),method='bounded');temp=float(np.exp(fit.x))
    prob=expit(logits/temp).astype(np.float32)
    balanced=threshold_at_fpr(y,prob,.05);tight=threshold_at_fpr(y,prob,.02)
    precision,recall,cuts=precision_recall_curve(y,prob)
    viable=(precision[:-1]>=.7)&(cuts>=tight)&(recall[:-1]>0)
    conservative=float(cuts[np.flatnonzero(viable)[0]]) if viable.any() else None
    out={'temperature':temp,'threshold_balanced':balanced,'threshold_conservative':conservative,
         'checkpoint_sha256':sha256_file(folder/'best.pt'),
         'goals':{'balanced_fpr_max':.05,'conservative_fpr_max':.02,'conservative_precision_min':.7},
         'goals_origin':'project engineering operating points, not literature-defined success standards',
         'fit_source':'val_cal: disjoint complete flights, uniform label-independent 4096/region bank; no test',
         'raw':binary_metrics(y,expit(logits),.5),'balanced':binary_metrics(y,prob,balanced),
         'conservative':binary_metrics(y,prob,conservative) if conservative is not None else None,
         'brier_before':float(np.mean((expit(logits)-y)**2)),'brier_after':float(np.mean((prob-y)**2)),
         'warning_policy':'single above-threshold decision; fresh bidirectional measurements <=2s and positive observed margin for proactive warning; no undocumented debounce',
         'no_claim':'temperature scaling cannot improve AP; does not guarantee held-out precision/FPR goals'}
    np.savez_compressed(folder/'val_cal_scores.npz',y=y,logits=logits,probability=prob)
    write_json(path,out);return out


def banks():
    sel=read_json(RESULT/'selection.json');manifest,p=collection()
    test=[r for r in manifest['records'] if r['split']=='test']
    materialize('test8',test,p,8192,2414)
    # 공간 추가 변형은 공통 시험 인덱스의 같은 과거 창에 별도 입력만 붙입니다.
    from usable.spatial_data import materialize_spatial_bank
    if any('beacon_map' in k for k in sel.get('variants',[])):materialize_spatial_bank('test8')
    configure(11,2);results={}
    names=[f'{k}_s11' for k in sel.get('variants',list(VARIANTS))]+[f"{sel['selected_variant']}_s{s}" for s in (22,33)]
    for name in names:
        folder=RUNS/name;model,state,meta=load(folder);cal=calibrate(folder);bank=bank_for_state('test8',state)
        t=time.perf_counter();logits=predict_bank(model,bank);prob=expit(logits[:,0]/cal['temperature']).astype(np.float32)
        np.save(folder/'test_bank_probability.npy',prob)
        out={'run':name,'calibration':cal,'balanced':grouped(bank.selected[:,4],prob,bank.region,cal['threshold_balanced']),
             'conservative':grouped(bank.selected[:,4],prob,bank.region,cal['threshold_conservative']) if cal['threshold_conservative'] is not None else None,
             'parameters':meta['parameters'],'inference_s':time.perf_counter()-t,'test_samples':len(bank)}
        write_json(folder/'bank_evaluation.json',out);results[name]=out
        print(name,'test AP',out['balanced']['all']['average_precision'],flush=True)
    # 측정된 양방향 SINR 하나만 사용하는 단순 기준. 보정된 확률이라는 의미는 없습니다.
    raw=Bank('test8',list(range(48)));v=Bank('val_cal',list(range(48)))
    def baseline(b):
        q=expit(-np.asarray(b.values[:,-1,40])/3).astype(np.float32);q[~b.mask[:,-1,40]]=.5;return q
    vp,tp=baseline(v),baseline(raw);cut=threshold_at_fpr(v.selected[:,4],vp,.05)
    simple={'threshold':cut,'balanced':grouped(raw.selected[:,4],tp,raw.region,cut),'score_is_calibrated_probability':False}
    map_baseline=None
    if (PREPARED/'test8/spatial_values.npy').exists():
        def geometry_score(b,name):
            z=np.load(PREPARED/name/'spatial_values.npy',mmap_mode='r');zm=np.load(PREPARED/name/'spatial_mask.npy',mmap_mode='r')
            # 간섭 일정 가정 + 등속 위치 외삽만 사용한 비학습 기준. 3은 단조 점수 변환 척도.
            proxy=np.asarray(b.values[:,-1,40])-np.maximum(np.asarray(z[:,-1,4]),0)
            prob=expit(-proxy/3).astype(np.float32);valid=b.mask[:,-1,40]&zm[:,-1,4];prob[~valid]=.5
            return prob
        vp,tp=geometry_score(v,'val_cal'),geometry_score(raw,'test8');cut=threshold_at_fpr(v.selected[:,4],vp,.05)
        map_baseline={'threshold':cut,'balanced':grouped(raw.selected[:,4],tp,raw.region,cut),
                      'definition':'score from observed minimum SINR minus positive 5s constant-velocity map loss change; no learning; not calibrated probability'}
    out={'selection':sel,'results':results,'current_sinr_baseline':simple,'geometry_extrapolation_baseline':map_baseline,
         'test_geography_status':p['test_status'],'sample_bank':'same label-independent 8192/region; region-balanced, not full-flight prevalence'}
    write_json(RESULT/'bank_comparison.json',out);return out


@torch.inference_mode()
def predict_episode(model,state,r,index,batch=512):
    x,m=load_observation(r);features=np.asarray(state['feature_indices'],int)
    if max(features)>=48:
        with np.load(ROOT/r['path']/'spatial_observations.npz') as z:
            x=np.concatenate((x,z['features']),axis=-1);m=np.concatenate((m,z['mask']),axis=-1)
    mean=np.asarray(state['normalizer']['mean'],np.float32)[features];std=np.asarray(state['normalizer']['std'],np.float32)[features]
    out=np.empty(len(index),np.float32)
    for first in range(0,len(index),batch):
        k,i,j=index[first:first+batch].T;t=k[:,None]+np.arange(-39,1)
        xx=x[t,i[:,None],j[:,None]][:,:,features];mm=m[t,i[:,None],j[:,None]][:,:,features]
        xx=np.where(mm,np.clip((xx-mean)/std,-8,8),0).astype(np.float32)
        out[first:first+len(k)]=model(torch.from_numpy(xx),torch.from_numpy(mm))[:,0].numpy()
    return out


def dense(name):
    configure(11,2);folder=RUNS/name;model,state,_=load(folder);cal=calibrate(folder)
    manifest,p=collection();all_y=[];all_p=[];all_region=[];events=[];events_tight=[];episodes=[]
    cache=folder/'dense';cache.mkdir(exist_ok=True)
    for r in [x for x in manifest['records'] if x['split']=='test']:
        ix=index_episode(r,p);take=np.flatnonzero(ix['current_available']);idx=ix['index'][take];y=ix['y'][take]
        file=cache/(r['id']+'.npz')
        if file.exists():
            with np.load(file) as z:
                if str(z['checkpoint_sha256'])!=sha256_file(folder/'best.pt'):raise ValueError('저장된 예측과 가중치 불일치')
                np.testing.assert_array_equal(idx,z['index']);np.testing.assert_array_equal(y,z['y']);prob=z['probability']
        else:
            prob=expit(predict_episode(model,state,r,idx)/cal['temperature']).astype(np.float32)
            np.savez_compressed(file,index=idx,y=y,probability=prob,checkpoint_sha256=sha256_file(folder/'best.pt'))
        all_y.append(y);all_p.append(prob);all_region.extend([r['region']]*len(y))
        events.extend(event_rows(r,idx,prob,cal['threshold_balanced'],p))
        if cal['threshold_conservative'] is not None:events_tight.extend(event_rows(r,idx,prob,cal['threshold_conservative'],p))
        episodes.append({'id':r['id'],'region':r['region'],'kind':r['metadata']['kind'],'samples':len(y),
                         'metrics':binary_metrics(y,prob,cal['threshold_balanced'])})
        print('dense',name,r['id'],len(y),flush=True)
    y=np.concatenate(all_y);prob=np.concatenate(all_p);regions=np.array(all_region)
    out={'run':name,'samples':len(y),'balanced':grouped(y,prob,regions,cal['threshold_balanced']),
         'conservative':grouped(y,prob,regions,cal['threshold_conservative']) if cal['threshold_conservative'] is not None else None,
         'balanced_events':summarize_events(events),'conservative_events':summarize_events(events_tight) if events_tight else None,
         'episodes':episodes,'scope':'all available eligible link windows, 48 flights, no failed-flight exclusion; geography previously inspected',
         'truth_use':'current true availability restricts this OFFLINE evaluation cohort only; never an inference input',
         'event_definition':'directed link unavailability >=1s; bidirectional packet success >=0.9 defines up; not physical packet outcome or network partition',
         'runtime_warning_gate_in_metrics':False}
    write_json(folder/'dense_evaluation.json',out);write_json(folder/'dense_events.json',events)
    return out


if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('stage',choices=['select','bank','dense']);pa.add_argument('--run');a=pa.parse_args()
    if a.stage=='select':print(json.dumps(freeze_selection(),indent=2))
    elif a.stage=='bank':banks()
    else:dense(a.run)
