"""검증에서 정한 오경보 예산·단순 비교군·사건 선행시간을 함께 비교한다.

물리 채널과 신경망 학습은 그대로 유지한다. 현재 가용 진실값은 사후 평가 집단만
나누며 모델 입력으로 사용하지 않는다. 시험으로 경고 기준을 다시 정하지 않는다.
"""
from pathlib import Path
from itertools import combinations
from html import escape
import csv
import json
import time
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve
from .forecast_benchmark import read_config, experiment_contract, fingerprint, load_arrays, write_csv
from .forecast_evaluation import binary_metrics, validation_threshold, evaluation_context, event_metrics, grouped_metrics
from .forecast_models import MODEL_NAMES
from .regions import read_json, write_json, sha256_file

ORDER=['gru','tcn','transformer','logistic_summary','ack_failure','constant']
LABEL={'gru':'GRU','tcn':'TCN','transformer':'Transformer','logistic_summary':'Logistic summary',
       'ack_failure':'ACK failure rule','constant':'Constant prior'}
COLOR=dict(zip(ORDER,['#2563eb','#059669','#e15b2d','#7c3aed','#64748b','#a1a1aa']))
OPS={'fpr05':'검증 FPR 5% 이하','fpr10':'검증 FPR 10% 이하','f1':'검증 F1 최대화'}
REGION={'teheran':'테헤란로','gangnam':'강남역','hongdae':'홍대','bundang':'분당','sangam_dmc':'상암 DMC'}
METRICS=['macro_region_ap','average_precision','auroc','precision','recall','false_positive_rate','brier',
         'strictly_early_recall','event_recall_lead_ge_1s','event_recall_lead_ge_2s','event_recall_lead_ge_3s','lead_time_median_s']


def fpr_budget_threshold(y,p,budget,split='val'):
    """검증 FPR 상한 내 최대 재현율. 동률은 더 낮은 FPR·더 높은 기준을 선택."""
    y,p=np.asarray(y),np.asarray(p,float)
    if split!='val': raise ValueError('경고 기준은 val에서만 정할 수 있습니다.')
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not 0<y.sum()<len(y):
        raise ValueError('같은 크기의 양성·음성 검증 자료가 필요합니다.')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.any((y!=0)&(y!=1)) or not 0<=budget<=1:
        raise ValueError('정답·확률·예산 범위 오류')
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    take=np.flatnonzero(fpr<=budget+1e-12)
    take=take[np.isclose(tpr[take],tpr[take].max(),rtol=0,atol=1e-12)]
    take=take[np.isclose(fpr[take],fpr[take].min(),rtol=0,atol=1e-12)]
    # 같은 확률의 표본을 쪼개지 않는다. 모든 경고를 끄는 경우도 명시적으로 허용한다.
    return min(float(thresholds[take[0]]),1.+1e-7)


def summary_features(x):
    """신경망과 같은 과거 창의 마지막 값·평균·표준편차·변화량만 사용."""
    x=np.asarray(x,np.float32)
    if x.ndim!=3 or not np.isfinite(x).all(): raise ValueError('유한한 [B,T,F] 입력이 필요합니다.')
    return np.concatenate([x[:,-1],x.mean(1),x.std(1),x[:,-1]-x[:,0]],1)


def ack_failure_score(arrays):
    """과거 송신의 ACK 실패 비율. +1/+2는 과제 비교군의 Beta(1,1) 평활."""
    d=arrays['data']; x=arrays['x']; f=x.shape[-1]//2
    a=d.meta['feature_names'].index('tx_attempts_in_interval')
    s=d.meta['feature_names'].index('ack_successes_in_interval')
    attempts=np.maximum((x[:,:,a]*d.std[a]+d.mean[a])*(x[:,:,f+a]>.5),0).sum(1)
    success=np.maximum((x[:,:,s]*d.std[s]+d.mean[s])*(x[:,:,f+s]>.5),0).sum(1)
    return np.clip((np.maximum(attempts-success,0)+1)/(attempts+2),0,1)


def prediction_file(run,split,arrays,context):
    path=run/f'{split}_predictions.csv'
    with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if len(rows)!=len(arrays['y']): raise ValueError('전체 창의 예측 파일이 필요합니다.')
    p=np.array([float(r['risk']) for r in rows])
    for j,r in enumerate(rows):
        epno,end,src,dst=map(int,arrays['data'].index[arrays['positions'][j]])
        ep=arrays['data'].collection['episodes'][epno]
        expected_time=float(arrays['data']._episode(epno)[2][end])
        if (r['episode_id']!=ep['id'] or r['region']!=ep['region'] or int(r['source'])!=src
            or int(r['destination'])!=dst or int(r['future_outage_label'])!=int(arrays['y'][j])
            or abs(float(r['input_end_s'])-expected_time)>1e-7
            or int(r['current_available_evaluation_only'])!=int(context.current_available[j])):
            raise ValueError('예측 CSV와 데이터셋의 행 출처가 다릅니다.')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('예측 확률 오류')
    return p,sha256_file(path)


def operations(y,p,context):
    up=context.current_available
    return {'f1':validation_threshold(y,p,up),
            'fpr05':fpr_budget_threshold(y[up],p[up],.05),
            'fpr10':fpr_budget_threshold(y[up],p[up],.10)}


def aggregate(rows,keys,metrics):
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,group in groups.items():
        row=dict(zip(keys,key))|{'runs':len(group)}
        for m in metrics:
            v=[r[m] for r in group if r.get(m) is not None]
            row[m]=float(np.mean(v)) if v else None
            row[m+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else None
        out.append(row)
    return out


def evaluate(record,split,a,c):
    p=record['p'][split]; up=c.current_available; result=[]
    for op,t in record['thresholds'].items():
        g=grouped_metrics(a['y'],p,c,t)
        e,events=event_metrics(c,p,t)
        eligible=[r for r in events if r['eligible']]
        row={'model':record['model'],'seed':record['seed'],'split':split,'operation':op,'threshold':t,
            **g['currently_available'],'macro_region_ap':g['macro_available_region_ap'],
            'strictly_early_recall':e['strictly_early_recall'],
            'lead_time_median_s':e['lead_time_median_s'],'eligible_events':len(eligible)}
        for seconds in (1,2,3):
            row[f'event_recall_lead_ge_{seconds}s']=(sum(r['lead_time_s'] is not None and
                r['lead_time_s']>=seconds-1e-8 for r in eligible)/len(eligible)) if eligible else None
        result.append(row)
    return result


def pairwise(records,split,a,c):
    rows=[]; y=a['y'][c.current_available].astype(bool)
    for a_name,b_name in combinations(MODEL_NAMES,2):
        for seed in sorted({r['seed'] for r in records if r['model'] in MODEL_NAMES}):
            ra=next(r for r in records if r['model']==a_name and r['seed']==seed)
            rb=next(r for r in records if r['model']==b_name and r['seed']==seed)
            pa,pb=ra['p'][split][c.current_available],rb['p'][split][c.current_available]
            cor=float(np.corrcoef(pa,pb)[0,1]) if pa.std()>0 and pb.std()>0 else None
            for op in OPS:
                da,db=pa>=ra['thresholds'][op],pb>=rb['thresholds'][op]
                rows.append({'split':split,'pair':a_name+' vs '+b_name,'seed':seed,'operation':op,
                    'risk_correlation':cor,'decision_disagreement':float(np.mean(da!=db)),
                    'both_miss_fraction':float(np.mean((~da&~db)[y])) if y.any() else None,
                    'a_only_detect_fraction':float(np.mean((da&~db)[y])) if y.any() else None,
                    'b_only_detect_fraction':float(np.mean((~da&db)[y])) if y.any() else None})
    return rows


def figure_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
        'axes.titleweight':'bold','savefig.facecolor':'white'})
    return plt


def savefig(fig,folder,name,plt):
    folder.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    path=folder/(name+'.svg'); fig.savefig(path,bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=150,bbox_inches='tight'); plt.close(fig)
    svg=path.read_text()
    # 여러 SVG를 한 HTML에 넣어도 그림 내부 참조 ID가 충돌하지 않도록 한다.
    ids=set(re.findall(r'\bid="([^"]+)"',svg))
    for old in sorted(ids,key=len,reverse=True):
        new=name+'__'+old
        svg=svg.replace('id="'+old+'"','id="'+new+'"').replace('url(#'+old+')','url(#'+new+')')
        svg=svg.replace('href="#'+old+'"','href="#'+new+'"')
    path.write_text(svg)
    return svg[svg.index('<svg'):]


def learning_plot(histories,chance,folder,plt):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name in MODEL_NAMES:
        hs=[h for (model,seed),h in histories.items() if model==name]
        for ax,key,title in zip(axes,['train_bce','val_bce','val_macro_available_region_ap'],
                                ['Training loss','Validation loss','Validation AP: current links up']):
            for h in hs: ax.plot([r['epoch'] for r in h],[r[key] for r in h],color=COLOR[name],alpha=.22,lw=1)
            common=min(len(h) for h in hs)
            values=np.array([[r[key] for r in h[:common]] for h in hs])
            ax.plot(np.arange(1,common+1),values.mean(0),color=COLOR[name],lw=2.2,label=LABEL[name])
            ax.set(title=title,xlabel='Epoch',ylim=(0,None))
    axes[0].set_ylabel('Binary cross entropy'); axes[1].set_ylabel('Binary cross entropy')
    axes[2].set_ylabel('Mean of regional AP')
    axes[2].axhline(chance,color='#6b7280',ls='--',label=f'No-skill reference {chance:.3f}')
    for ax in axes: ax.legend(fontsize=8)
    return savefig(fig,folder,'learning_overlay',plt)


def curve_plot(records,split,a,c,folder,plt):
    fig,axes=plt.subplots(1,3,figsize=(15,4.2)); up=c.current_available; y=a['y'][up]
    for model in ORDER[:-1]:
        selected=[r for r in records if r['model']==model]
        aps=[binary_metrics(y,r['p'][split][up])['average_precision'] for r in selected]
        for j,r in enumerate(selected):
            p=r['p'][split][up]; pr,re,_=precision_recall_curve(y,p); fp,tp,_=roc_curve(y,p)
            ix=np.unique(np.linspace(0,len(re)-1,min(500,len(re))).astype(int))
            ir=np.unique(np.linspace(0,len(fp)-1,min(500,len(fp))).astype(int))
            alpha=.4 if len(selected)>1 else .85
            axes[0].plot(re[ix],pr[ix],color=COLOR[model],alpha=alpha,lw=1.5,
                label=f'{LABEL[model]} AP={np.mean(aps):.3f}' if j==0 else None)
            for ax in axes[1:]: ax.plot(fp[ir],tp[ir],color=COLOR[model],alpha=alpha,lw=1.5,
                label=LABEL[model] if j==0 else None)
    axes[0].axhline(float(y.mean()),ls='--',color='#9ca3af',label=f'Prevalence {y.mean():.3f}')
    axes[0].set(title=f'{split.upper()} precision-recall',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    for ax in axes[1:]:
        ax.plot([0,1],[0,1],ls='--',lw=1,color='#9ca3af')
        ax.set(xlabel='False positive rate',ylabel='Recall',ylim=(0,1))
    axes[1].set(title='ROC: full range',xlim=(0,1)); axes[2].set(title='ROC: low false-alarm range',xlim=(0,.2))
    for ax in axes: ax.legend(fontsize=7,loc='best')
    return savefig(fig,folder,f'{split}_pr_roc',plt)


def event_plot(rows,split,folder,plt):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,op in zip(axes,['fpr05','fpr10']):
        for model in ORDER[:-1]:
            r=next(r for r in rows if r['model']==model and r['split']==split and r['operation']==op)
            keys=['strictly_early_recall','event_recall_lead_ge_1s','event_recall_lead_ge_2s','event_recall_lead_ge_3s']
            ax.plot([0,1,2,3],[np.nan if r[k] is None else r[k] for k in keys],'o-',color=COLOR[model],label=LABEL[model],ms=4)
        ax.set(title=f'{split.upper()}: validation FPR <= {5 if op=="fpr05" else 10}%',
            xlabel='Required lead (s; 0 means strictly >0)',ylabel='Fraction of eligible events warned',xticks=[0,1,2,3],ylim=(0,1))
        ax.legend(fontsize=8)
    return savefig(fig,folder,f'{split}_event_lead',plt)


def region_plot(rows,split,folder,plt):
    selected=[r for r in rows if r['split']==split]; regions=sorted({r['region'] for r in selected})
    values=np.array([[next(r['average_precision'] for r in selected if r['model']==m and r['region']==g)
                      for g in regions] for m in ORDER],float)
    fig,ax=plt.subplots(figsize=(8,4.8)); im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(len(regions)),xticklabels=regions,yticks=range(len(ORDER)),
        yticklabels=[LABEL[m] for m in ORDER],title=f'{split.upper()} regional AP: current links up')
    for i in range(len(ORDER)):
        for j in range(len(regions)):
            v=values[i,j]; ax.text(j,i,'n/a' if not np.isfinite(v) else f'{v:.3f}',ha='center',va='center',
                color='white' if v>.55 else '#17233b')
    ax.grid(False); fig.colorbar(im,ax=ax,label='Average precision')
    return savefig(fig,folder,f'{split}_regional_ap',plt)


def timeline_plot(records,split,a,c,folder,plt):
    events=sorted([e for e in c.events if len(e['candidate_positions']) and e['start_s']>=60],
                  key=lambda e:(e['region'],e['episode_id'],e['source'],e['destination'],e['start_s']))
    if not events: return '',None
    event=events[0]  # 모델의 성공 여부를 보지 않고 정하는 사례 선택 규칙.
    index=a['data'].index[a['positions']]
    pos=np.flatnonzero((c.episode_ids==event['episode_id'])&(index[:,2]==event['source'])&(index[:,3]==event['destination']))
    times=a['data']._episode(int(index[pos[0],0]))[2][index[pos,1]]
    near=(times>=event['start_s']-20)&(times<=event['start_s']+15); pos,times=pos[near],times[near]
    fig,axes=plt.subplots(5,1,figsize=(11,8.2),sharex=True)
    axes[0].step(times,a['y'][pos],where='post',color='#e15b2d',label='Future-outage label')
    axes[0].step(times,~c.current_available[pos],where='post',color='#64748b',alpha=.7,label='Already unavailable')
    axes[0].legend(fontsize=8,loc='upper right'); axes[0].set_ylim(-.05,1.1)
    for ax,model in zip(axes[1:],['gru','tcn','transformer','logistic_summary']):
        r=next(r for r in records if r['model']==model)
        ax.plot(times,r['p'][split][pos],color=COLOR[model],label=f'{LABEL[model]} seed={r["seed"]}')
        ax.axhline(r['thresholds']['f1'],ls='--',color=COLOR[model],label='Validation F1 threshold')
        ax.set(ylim=(0,1),ylabel='Risk'); ax.legend(fontsize=8,loc='upper right')
    for ax in axes: ax.axvline(event['start_s'],color='#dc2626',lw=1)
    axes[-1].set_xlabel('Simulation time (s); red line = actual outage onset')
    axes[0].set_title(f"{event['region']}: link {event['source']} -> {event['destination']} | first eligible example")
    meta={k:v for k,v in event.items() if not k.startswith('candidate_')}
    return savefig(fig,folder,f'{split}_timeline',plt),meta


def number(v,pct=False):
    return '—' if v is None else (f'{100*v:.1f}%' if pct else f'{v:.3f}')


def table(rows,columns):
    head=''.join('<th>'+escape(label)+'</th>' for k,label,pct in columns); body=[]
    for r in rows:
        cells=[]
        for k,label,pct in columns:
            v=r.get(k)
            if k=='model': text=LABEL.get(v,v)
            elif isinstance(v,str): text=REGION.get(v,v)
            elif k in ('runs','seed','parameters','eligible_events'): text=str(v) if v is not None else '—'
            else:
                text=number(v,pct)
                if r.get(k+'_std') is not None: text+=' ± '+number(r[k+'_std'],pct)
            cells.append('<td>'+escape(text)+'</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def detailed_comparison(config_path,include_test=False):
    cfg,folder,out=read_config(config_path)
    if cfg['stage']=='smoke': raise ValueError('상세 비교는 전체 창의 pilot/final에서 실행합니다.')
    token=fingerprint(experiment_contract(cfg,folder)); dest=out/'detailed'; dest.mkdir(parents=True,exist_ok=True)
    records=[]; histories={}
    for name in MODEL_NAMES:
        for seed in cfg['seeds']:
            run_folder=out/name/f'seed_{seed}'; run=read_json(run_folder/'run.json')
            if not run.get('complete') or run['experiment_sha256']!=token: raise ValueError('모든 모델·시드의 같은 조건 학습을 완료하십시오.')
            if sha256_file(run_folder/'best.pt')!=run['checkpoint_sha256']: raise ValueError('체크포인트가 변경되었습니다.')
            records.append({'model':name,'seed':seed,'folder':run_folder,'run':run,'p':{}})
            with (run_folder/'history.csv').open(encoding='utf-8-sig') as f:
                histories[(name,seed)]=[{k:float(v) for k,v in r.items() if v} for r in csv.DictReader(f)]
    print('기준 모델은 train, 경고 기준은 val에서만 정합니다.',flush=True)
    train=load_arrays(folder,'train'); val=load_arrays(folder,'val'); vc=evaluation_context(val['data'],val['positions'])
    started=time.perf_counter(); scaler=StandardScaler(); tx=scaler.fit_transform(summary_features(train['x']))
    lr=LogisticRegression(C=1.,max_iter=1000,random_state=2026,solver='lbfgs'); lr.fit(tx,train['y'])
    prior=float(train['y'].mean())
    refs={'logistic_summary':lr.predict_proba(scaler.transform(summary_features(val['x'])))[:,1],
          'ack_failure':ack_failure_score(val),'constant':np.full(len(val['y']),prior)}
    write_json(dest/'baseline_model.json',{'fit_split':'train','C':1.,'solver':'lbfgs','n_iter':lr.n_iter_.tolist(),
        'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
        'coef':lr.coef_.tolist(),'intercept':lr.intercept_.tolist(),'prior':prior,
        'summary_rule':'last, mean, std(ddof=0), last-first of same packed past input',
        'ack_rule':'(failed attempts + 1)/(attempts + 2)','fit_seconds':time.perf_counter()-started})
    for name,p in refs.items(): records.append({'model':name,'seed':0,'run':None,'p':{'val':p}})
    protocol={'format':'forecast-diagnostic-protocol-v1','training_experiment_sha256':token,
        'source_sha256':sha256_file(Path(__file__)),'threshold_fit_split':'val','fpr_budgets':[.05,.10],
        'minimum_warning_lead_s':[1,2,3],'thresholds':{},'validation_prediction_sha256':{},
        'composite_score_used':False,'current_availability_is_not_an_input':True,'test_used_to_tune':False}
    for r in records:
        key=f"{r['model']}/seed_{r['seed']}"
        if r['run']:
            r['p']['val'],sha=prediction_file(r['folder'],'val',val,vc); protocol['validation_prediction_sha256'][key]=sha
        r['thresholds']=operations(val['y'],r['p']['val'],vc); protocol['thresholds'][key]=r['thresholds']
    frozen=dest/'protocol_frozen_before_test.json'
    if frozen.exists() and read_json(frozen)!=protocol: raise ValueError('기존 고정 비교 규칙과 다릅니다. 새 output을 사용하십시오.')
    write_json(frozen,protocol)  # 시험 입력·정답을 읽기 전에 고정한다.
    data={'val':val}; contexts={'val':vc}
    if include_test:
        a=load_arrays(folder,'test'); c=evaluation_context(a['data'],a['positions']); data['test']=a; contexts['test']=c
        reference={'logistic_summary':lr.predict_proba(scaler.transform(summary_features(a['x'])))[:,1],
            'ack_failure':ack_failure_score(a),'constant':np.full(len(a['y']),prior)}
        for r in records:
            if r['run']:
                m=read_json(r['folder']/'test_metrics.json')
                if m.get('checkpoint_sha256')!=r['run']['checkpoint_sha256'] or not m.get('test_did_not_change_model_or_threshold'):
                    raise ValueError('고정 모델의 시험 결과가 필요합니다.')
                r['p']['test'],_=prediction_file(r['folder'],'test',a,c)
            else: r['p']['test']=reference[r['model']]
    rows=[]; regions=[]; episodes=[]; pairs=[]; strata=[]
    for split,a in data.items():
        c=contexts[split]; pairs.extend(pairwise(records,split,a,c))
        for r in records:
            p=r['p'][split]; rows.extend(evaluate(r,split,a,c))
            g=grouped_metrics(a['y'],p,c,r['thresholds']['f1'])
            for region,m in g['by_region'].items(): regions.append({'model':r['model'],'seed':r['seed'],'split':split,'region':region,**m['currently_available']})
            for episode,m in g['by_episode'].items(): episodes.append({'model':r['model'],'seed':r['seed'],'split':split,'episode':episode,**m['currently_available']})
            age=a['data'].meta['feature_names'].index('seconds_since_last_ack'); f=a['x'].shape[-1]//2
            known=a['x'][:,-1,f+age]>.5; ages=a['x'][:,-1,age]*a['data'].std[age]+a['data'].mean[age]
            for label,take in {'ACK age <=1s':known&(ages<=1),'ACK age 1-5s':known&(ages>1)&(ages<=5),
                               'ACK age >5s':known&(ages>5),'No ACK yet':~known}.items():
                take=take&c.current_available
                strata.append({'model':r['model'],'seed':r['seed'],'split':split,'stratum':label,
                    **binary_metrics(a['y'][take],p[take],r['thresholds']['f1'])})
    averages=aggregate(rows,['split','model','operation'],METRICS)
    regional=aggregate(regions,['split','model','region'],['average_precision','recall','precision','false_positive_rate'])
    paired=aggregate(pairs,['split','pair','operation'],['risk_correlation','decision_disagreement','both_miss_fraction','a_only_detect_fraction','b_only_detect_fraction'])
    for name,rs in {'operating_points_by_seed.csv':rows,'operating_points_mean_std.csv':averages,
        'regions_by_seed.csv':regions,'regions_mean_std.csv':regional,'episodes_by_seed.csv':episodes,
        'pairwise_disagreement_by_seed.csv':pairs,'pairwise_disagreement_mean_std.csv':paired,'ack_freshness_by_seed.csv':strata}.items(): write_csv(dest/name,rs)
    chance={s:float(a['y'][contexts[s].current_available].mean()) for s,a in data.items()}
    macro_chance=float(np.mean([val['y'][(vc.regions==g)&vc.current_available].mean() for g in sorted(set(vc.regions))]))
    plt=figure_setup(); figures=dest/'figures'; learning=learning_plot(histories,macro_chance,figures,plt)
    graphs={}; examples={}
    for split,a in data.items():
        c=contexts[split]; trace,examples[split]=timeline_plot(records,split,a,c,figures,plt)
        graphs[split]={'pr':curve_plot(records,split,a,c,figures,plt),'events':event_plot(averages,split,figures,plt),
                      'regions':region_plot(regional,split,figures,plt),'trace':trace}
    result={'format':'forecast-detailed-comparison-v1','stage':cfg['stage'],'protocol_sha256':sha256_file(frozen),
        'training_experiment_sha256':token,'test_evaluated':include_test,'neural_runs':3*len(cfg['seeds']),
        'seeds_per_model':len(cfg['seeds']),'dataset_splits':val['data'].meta['splits'],
        'design':val['data'].collection.get('config',{}).get('design'),
        'chance_currently_available':chance,'aggregates':averages,'regions':regional,'pairs':paired,'timeline_examples':examples,
        'final_model_selected':False,'repositioning_evaluated':False,
        'scope':'동일 배치·지도·통신 모델의 예측 비교. 시드 편차는 독립 도시 신뢰구간이 아님.'}
    write_json(dest/'summary.json',result)
    (dest/'comparison_detailed.html').write_text(render_html(result,records,graphs,learning),encoding='utf-8')
    basic=out/'comparison.html'
    if basic.exists():
        text=basic.read_text(); notice='<p class="note" id="detailed-link"><a href="detailed/comparison_detailed.html">새 상세 비교: 같은 오경보 예산 · PR 곡선 · 지역 · 선행시간 · 단순 기준 모델</a></p>'
        if 'id="detailed-link"' not in text: basic.write_text(text.replace('<main>','<main>'+notice,1),encoding='utf-8')
    print(json.dumps({'html':str(dest/'comparison_detailed.html'),'test_evaluated':include_test,'neural_runs':result['neural_runs']},ensure_ascii=False),flush=True)
    return result


def render_html(s,records,graphs,learning):
    d=s.get('design')
    scope='설정 파일에 명시한 고정 배치의 시뮬레이션 기록'
    if d:
        formation={'v':'V','line':'일렬','square':'사각','circle':'원형'}.get(d['formation'],d['formation'])
        scope=(f"전체 {d['n_drones']}대(수색 {d['n_drones']-d['n_relays']}·중계 {d['n_relays']}), "
               f"수색 {formation} 대형, 간격 {d['spacing_m']:g}m의 고정 비행 기록")
    columns=[('model','모델',False),('runs','반복 수',False),('macro_region_ap','지역 평균 AP ↑',False),
        ('average_precision','통합 AP ↑',False),('recall','장애 재현율 ↑',True),('precision','경고 정밀도 ↑',True),
        ('false_positive_rate','실제 FPR ↓',True),('event_recall_lead_ge_1s','1초 이상 사전 경고 ↑',True),('brier','Brier ↓',False)]
    blocks=[]
    for split,g in graphs.items():
        parts=[]; pparts=[]
        for op in OPS:
            rows=[r for r in s['aggregates'] if r['split']==split and r['operation']==op]; rows.sort(key=lambda r:ORDER.index(r['model']))
            parts.append(f'<div class="operating" data-op="{op}"><h3>{OPS[op]}</h3>'+table(rows,columns)+'</div>')
            rows=[r for r in s['pairs'] if r['split']==split and r['operation']==op]
            pparts.append(f'<div class="operating" data-op="{op}">'+table(rows,[('pair','모델 A vs B',False),('risk_correlation','위험도 상관계수',False),
                ('decision_disagreement','서로 다른 경고 판단',True),('both_miss_fraction','둘 다 놓친 장애',True),
                ('a_only_detect_fraction','A만 탐지',True),('b_only_detect_fraction','B만 탐지',True)])+'</div>')
        rt=[r for r in s['regions'] if r['split']==split]
        blocks.append(f'''<section class="split" data-split="{split}"><h2>{'검증: 별도 실행' if split=='val' else '시험: 예측기 학습에서 보류한 지역'}</h2>
        <p>현재 연결된 링크의 미래 장애 비율: <b>{s['chance_currently_available'][split]*100:.2f}%</b>. 통합 AP는 이 기준선과 함께 보십시오.</p>
        {''.join(parts)}<p class="small">±는 같은 데이터의 모델 초기화·학습 순서 시드 간 표준편차입니다. 도시 일반화의 신뢰구간이 아닙니다.
        오경보 상한은 검증에서만 맞췄으므로 시험의 실제 FPR을 함께 보십시오.</p>
        <h2>경고를 늘리면 장애를 얼마나 더 잡는가?</h2>{g['pr']}
        <p class="small">옅은 선은 각 학습 시드입니다. PR의 회색선은 장애 비율입니다. 오른쪽 ROC는 FPR 0~20% 확대입니다.
        그림만 최대 500점으로 줄였고 모든 지표는 전체 창으로 계산했습니다.</p>
        <h2>단절 전에 행동할 시간을 확보하는가?</h2>{g['events']}
        <p>0 지점은 선행시간이 엄격히 0초보다 큰 경고입니다. 1·2·3초는 비교용 여유이며 실제 재배치에 충분한지는 아직 평가하지 않았습니다.</p>
        <h2>지역별 차이</h2>{g['regions']}{table(rt,[('model','모델',False),('region','지역',False),('average_precision','AP ↑',False),
        ('recall','재현율 ↑',True),('precision','정밀도 ↑',True),('false_positive_rate','FPR ↓',True)])}
        <h2>실제로 같은 예측을 하는가?</h2>{''.join(pparts)}<p class="small">같은 번호의 시드끼리 비교했습니다.
        상관계수 1에 가까울수록 위험도 출력이 비슷합니다. 놓침·한 모델만 탐지의 분모는 실제 미래 장애 창입니다. 유의성 검정은 아닙니다.</p>
        <h2>같은 단절 시점의 위험도</h2>{g['trace']}<p class="small">지역·실행·드론 번호·시각순 첫 적격 사건입니다. 성공 여부로 사례를 고르지 않았습니다.
        맨 위에서 미래 장애 정답과 이미 단절된 상태를 구분합니다. 아래 점선은 모델별 검증 F1 경고 기준입니다.</p></section>''')
    costs=[]
    for model in MODEL_NAMES:
        rs=[r['run'] for r in records if r['model']==model]
        costs.append({'model':model,'parameters':rs[0]['parameters'],
            'training_minutes':float(np.mean([r['elapsed_training_s']/60 for r in rs])),
            'inference_p95_ms':float(np.mean([r['latency']['single_window_p95_ms'] for r in rs]))})
    cost=table(costs,[('model','모델',False),('parameters','파라미터 수',False),('training_minutes','평균 학습(분)',False),('inference_p95_ms','평균 단일 창 p95(ms)',False)])
    controls='<button data-target="val">검증: 테헤란로·강남역·홍대</button>'
    if s['test_evaluated']: controls+='<button data-target="test">시험: 분당·상암 DMC</button>'
    return '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>통신 장애 예측 · 세 모델 상세 비교</title><style>*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:#172a40;font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",sans-serif}
    header{background:#122b46;color:white;padding:38px max(24px,calc((100vw - 1240px)/2))}header p{color:#c4d5e6}h1{font-size:30px;line-height:1.4;margin:8px 0}
    main{max-width:1320px;margin:24px auto;padding:0 20px 48px}section,.card{background:white;border:1px solid #dbe4ed;border-radius:14px;padding:24px;margin-bottom:20px}
    h2{font-size:21px;margin:30px 0 16px;border-top:1px solid #dbe4ed;padding-top:24px}h3{font-size:17px}p,li{line-height:1.8}.small{font-size:13px;color:#5e7084}
    .note{border-left:4px solid #dd9e25;background:#fff7e5;padding:14px 18px}.controls{position:sticky;top:0;z-index:5;background:#eef3f7f2;padding:12px 0;display:flex;flex-wrap:wrap;gap:8px}
    button,select{padding:11px 16px;border:1px solid #b9cadd;background:white;border-radius:8px;font:inherit;color:#172a40;cursor:pointer}button.active{background:#2563eb;color:white;border-color:#2563eb}
    svg{width:100%;height:auto}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th{background:#edf3fa;color:#24486f}th,td{padding:12px 10px;text-align:right;border-bottom:1px solid #dbe4ed}th:first-child,td:first-child{text-align:left}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.split[hidden],.operating[hidden]{display:none}a{color:#1c5eb5}.formula{font-family:Georgia,serif;background:#f2f6fb;padding:12px;border-radius:8px}
    @media(max-width:720px){.grid{grid-template-columns:1fr}section,.card{padding:16px}header{padding:26px 20px}h1{font-size:25px}}</style></head><body>
    <header><p>UAV LINK FAILURE PREDICTION</p><h1>GRU · TCN · Transformer<br>어떤 조건에서 차이가 생기는가?</h1><p>같은 과거 기록 · 같은 정답 · 같은 지역 분할 · 반복 학습 · 종합 점수 없음</p></header><main>
    <div class="card note"><b>범위:</b> '''+escape(scope)+'''입니다. 예측에 따라 이동을 바꾸는 재배치 실험은 포함하지 않습니다.</div>
    <div class="grid"><div class="card"><h3>검증 AP는 정확도가 아닙니다</h3><p>위험도가 높은 순서대로 경고 대상을 늘릴 때 실제 장애가 목록 앞쪽에 얼마나 잘 모이는지 요약한 평균 정밀도입니다.</p>
    <div class="formula">AP = Σ (이번 재현율 − 직전 재현율) × 이번 정밀도</div><p class="small">위험도 순서의 정답이 장애·정상·장애·정상이면 AP=(1+2/3)/2≈0.833입니다.
    학습 중 지역 평균 AP는 각 검증 지역의 AP를 같은 비중으로 평균한 값입니다. 무정보 상수 예측의 AP는 해당 평가 집단의 장애 비율입니다.</p></div>
    <div class="card"><h3>왜 이전 그래프가 비슷했나요?</h3><p>2에포크 실행 점검과 0~1 고정 축으로는 학습과 모델 차이를 판단하기 어려웠습니다.
    지금은 전체 창 학습, 반복 시드, 공통 축의 곡선과 경고 예산을 비교합니다.</p><p class="small">모델이 계속 비슷하다면 그 결과도 그대로 보고합니다. 그래프 확대만으로 성능이 좋아지는 것은 아닙니다.</p></div></div>
    <div class="controls">'''+controls+'''<select id="operation"><option value="fpr05">검증 FPR 5% 이하</option><option value="fpr10">검증 FPR 10% 이하</option><option value="f1">검증 F1 최대화</option></select></div>
    '''+''.join(blocks)+'''<section><h2>학습이 실제로 진행되었는가?</h2>'''+learning+'''
    <p class="small">세로축은 0에서 시작합니다. 얇은 선은 개별 시드, 굵은 선은 모든 시드가 존재하는 구간의 평균입니다.
    조기 종료 뒤 값을 이어 그리지 않았습니다. 공통 설정의 비교이며 모델별 최적 하이퍼파라미터 탐색은 수행하지 않았습니다.</p>
    <h2>계산 비용</h2>'''+cost+'''<p class="small">현재 실행 PC의 측정입니다. 동시 작업의 영향이 있으므로 엄밀한 하드웨어 벤치마크나 실제 드론 처리시간으로 해석하지 않습니다.</p>
    <h2>선정에 사용할 순서</h2><ol><li>단순 기준 모델보다 나은지 봅니다.</li><li>허용 가능한 오경보 범위에서 몇 초 먼저 경고하는지 봅니다.</li><li>분당·상암 각각의 성능과 시드 편차를 봅니다.</li><li>차이가 작으면 계산량과 안정성을 함께 보고 선정합니다.</li></ol>
    <p class="note">5%·10%는 프로젝트 비교용 예산이며 논문이 보장하는 합격점이 아닙니다. 현재 가용성은 사후 진단 집단을 나누는 진실값이며 모델 입력이 아닙니다.
    두 시험 지역도 과거 대형 선정에는 사용했으므로 전체 시스템에서 완전히 미지인 지역 검증은 아닙니다.</p>
    <h2>단순 비교군</h2><ul><li>Constant prior: 학습 자료의 장애 비율만 출력합니다.</li><li>ACK failure rule: 최근 창의 `(실패+1)/(시도+2)`입니다. 프로젝트 비교용 평활 규칙이며 보정된 실제 확률이 아닙니다.</li>
    <li>Logistic summary: 같은 과거 입력의 마지막 값·평균·표준편차·변화량으로 학습한 C=1 분류기입니다. 표준화·가중치는 train에서만 적합했습니다.</li></ul>
    <h2>실제로 참고한 근거</h2><p><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">AP 공식 정의</a> ·
    <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432">Saito &amp; Rehmsmeier (2015): 불균형 자료의 PR 평가</a> ·
    <a href="https://scikit-learn.org/stable/modules/classification_threshold.html">별도 검증 자료의 경고 기준 조정</a> ·
    <a href="https://arxiv.org/abs/1809.04356">Fawaz et al. (2019): 초기화에 따른 성능 변동</a></p>
    <p class="small">평가 원칙의 근거입니다. 논문의 UAV 성능·안전 합격점을 가져온 것이 아닙니다. 10초 과거·5초 미래·1초 지속·경고 예산은 과제 설계값입니다.</p>
    <p>원시 수치는 ZIP의 동일 detailed 폴더에 있습니다: operating_points_by_seed.csv · operating_points_mean_std.csv · regions_by_seed.csv · episodes_by_seed.csv ·
    pairwise_disagreement_mean_std.csv · ack_freshness_by_seed.csv · summary.json · protocol_frozen_before_test.json.</p></section></main>
    <script>let split='''+json.dumps('test' if s['test_evaluated'] else 'val')+''';const buttons=[...document.querySelectorAll('[data-target]')];
    function update(){document.querySelectorAll('.split').forEach(e=>e.hidden=e.dataset.split!==split);buttons.forEach(b=>b.classList.toggle('active',b.dataset.target===split));const op=document.getElementById('operation').value;document.querySelectorAll('.operating').forEach(e=>e.hidden=e.dataset.op!==op)}
    buttons.forEach(b=>b.onclick=()=>{split=b.dataset.target;update()});document.getElementById('operation').onchange=update;update();</script></body></html>'''
