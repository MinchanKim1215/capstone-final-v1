"""저장된 비교 수치를 설명하는 표시 계층. 학습·경고 기준은 변경하지 않는다."""
from pathlib import Path
from html import escape
from .regions import write_json


def refresh_timeline_steps(summary,html_path,config_path):
    """관측 사이를 직선으로 잇지 않아 실제 없던 조기 경고를 그림에 만들지 않는다."""
    import csv
    import numpy as np
    from scipy.special import expit
    from .regions import read_json
    from .forecast_benchmark import read_config
    from .forecast_dataset import ForecastDataset
    from .forecast_models import packed_input
    from .forecast_diagnostics import summary_features, figure_setup, savefig
    cfg,folder,output=read_config(config_path)
    path=Path(html_path); detail=path.parent
    protocol=read_json(detail/'protocol_frozen_before_test.json')
    reference=read_json(detail/'baseline_model.json'); plt=figure_setup()
    text=path.read_text()
    for split,event in summary['timeline_examples'].items():
        if event is None:continue
        selected={}; positions=None
        for name in ['gru','tcn','transformer']:
            p=output/name/f"seed_{cfg['seeds'][0]}"/f'{split}_predictions.csv'
            with p.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            take=[i for i,r in enumerate(rows) if r['episode_id']==event['episode_id']
                and int(r['source'])==event['source'] and int(r['destination'])==event['destination']
                and event['start_s']-20<=float(r['input_end_s'])<=event['start_s']+15]
            chosen=[rows[i] for i in take]
            selected[name]=np.array([float(r['risk']) for r in chosen])
            if positions is None:
                positions=take
                times=np.array([float(r['input_end_s']) for r in chosen])
                y=np.array([int(r['future_outage_label']) for r in chosen])
                bad=1-np.array([int(r['current_available_evaluation_only']) for r in chosen])
        data=ForecastDataset(folder,split)
        samples=[data[i] for i in positions]
        x=packed_input(np.stack([r['x'] for r in samples]),np.stack([r['mask'] for r in samples]))
        z=(summary_features(x)-np.asarray(reference['scaler_mean']))/np.asarray(reference['scaler_scale'])
        selected['logistic_summary']=expit(z@np.asarray(reference['coef'])[0]+reference['intercept'][0])
        fig,axes=plt.subplots(5,1,figsize=(11,8.2),sharex=True)
        axes[0].step(times,y,where='post',color='#e15b2d',label='Future-outage label')
        axes[0].step(times,bad,where='post',color='#64748b',alpha=.7,label='Already unavailable')
        axes[0].set_ylim(-.05,1.1);axes[0].legend(fontsize=8,loc='upper right')
        colors=['#2563eb','#059669','#e15b2d','#7c3aed']
        for ax,(model,p),color in zip(axes[1:],selected.items(),colors):
            seed=cfg['seeds'][0] if model!='logistic_summary' else 0
            threshold=protocol['thresholds'][f'{model}/seed_{seed}']['f1']
            ax.step(times,p,where='post',marker='.',ms=3,color=color,label=f'{model} seed={seed}')
            ax.axhline(threshold,ls='--',color=color,label='Validation F1 threshold')
            ax.set(ylim=(0,1),ylabel='Risk');ax.legend(fontsize=8,loc='upper right')
        for ax in axes:ax.axvline(event['start_s'],color='#dc2626',lw=1)
        axes[-1].set_xlabel('Simulation time (s); dots are actual prediction times; red line = outage onset')
        axes[0].set_title(f"{event['region']}: link {event['source']} -> {event['destination']} | first eligible example")
        svg_path=detail/'figures'/f'{split}_timeline.svg'
        old=svg_path.read_text();old=old[old.index('<svg'):]
        svg=savefig(fig,detail/'figures',f'{split}_timeline',plt)
        text=text.replace(old,svg,1)
    path.write_text(text,encoding='utf-8')


def add_findings(summary,html_path,config_path=None):
    path=Path(html_path)
    if not summary.get('test_evaluated'):return
    if config_path is not None:refresh_timeline_steps(summary,path,config_path)
    names=['gru','tcn','transformer']
    def row(split,model,op='fpr05'):
        return next(r for r in summary['aggregates'] if r['split']==split and r['model']==model and r['operation']==op)
    validation=[row('val',m)['average_precision'] for m in names]
    test=[row('test',m)['average_precision'] for m in names]
    false_alarm=[row('test',m)['false_positive_rate'] for m in names]
    baseline=row('test','logistic_summary')['average_precision']
    messages=[
        f'세 신경망의 통합 AP: 검증 {min(validation):.3f}–{max(validation):.3f}, 시험 {min(test):.3f}–{max(test):.3f}. 학습 지역의 별도 실행 성능이 새 지역으로 그대로 유지되지 않았습니다.',
        f'요약 로지스틱 회귀의 시험 통합 AP는 {baseline:.3f}입니다. 복잡한 모델의 확실한 우월성을 AP만으로 주장할 수 없습니다.',
        f'검증 FPR 5% 이하 기준을 고정했을 때 세 신경망의 실제 시험 FPR은 {min(false_alarm)*100:.1f}–{max(false_alarm)*100:.1f}%입니다. 처음 보는 지역에서 기준의 전달 가능성도 함께 검토해야 합니다.',
        '현재 결과에서 최종 승자를 자동 선정하지 않았습니다. 다음 검증은 학습 환경 다양성·관측 신선도·입력 정보의 한계를 대상으로 하며, 같은 시험 자료로 반복 튜닝한 결과를 독립 시험이라고 부르지 않습니다.'
    ]
    write_json(path.parent/'interpretation.json',{'messages':messages,'model_or_threshold_changed':False,
        'timeline_draw_style':'steps-post: predictions start only at actual recorded inference times',
        'statistical_significance_claimed':False,'scope':'저장된 시드 평균의 기술적 비교. 원인 규명 실험은 아님.'})
    card='<div class="card note" id="key-findings"><h3>이번 결과에서 먼저 확인할 것</h3><ul>'+''.join('<li>'+escape(t)+'</li>' for t in messages)+'</ul></div>'
    text=path.read_text()
    if 'id="key-findings"' not in text:
        text=text.replace('<div class="controls">',card+'<div class="controls">',1)
        path.write_text(text,encoding='utf-8')
