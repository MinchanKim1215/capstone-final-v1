"""Verified scientific figures and a portable summary of prior-drop evidence."""
from pathlib import Path
from functools import lru_cache
import sys,json,base64,hashlib,html,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/precursor_drops_20261105'
NAMES={'abrupt':'급락형','gradual':'점진형','mixed':'혼합형','other':'기타형'}
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def img(p):return 'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode('ascii')


@lru_cache(maxsize=12)
def trace(region):
    with np.load(OUT/'traces'/(region+'.npz')) as z:
        return z['margin_db'].astype(float),z['network'],z['pulse']


def type_figure(s):
    fig,ax=plt.subplots(figsize=(13.2,6.7))
    fig.subplots_adjust(left=.12,right=.96,top=.77,bottom=.31)
    fig.text(.06,.94,'30초 이력에서 구분한 앞선 급락',fontsize=21,fontweight='bold')
    fig.text(.06,.87,'단절 3~30초 전 · 0.5초에 6dB 이상 감소 · 30초 이력이 완전한 사건만 동일 분모로 비교',fontsize=11.5,color='#526575')
    for j,(key,label,color) in enumerate([
        ('any_prior_jump','앞선 급락 전체 · 앞선 단절도 포함','#b5c2ca'),
        ('still_connected_jump','급락 전후에도 연결 유지','#3f9d99'),
        ('uninterrupted_to_onset_jump','그 급락부터 이번 단절 직전까지 계속 연결','#16675f')]):
        values=[]
        for i,k in enumerate(NAMES):
            t=s['types'][k];n=t['n30'];v=t['complete30'][key]
            values.append(v/n*100.)
            ax.text(v/n*100.+.8,i+(j-1)*.23,f'{v}/{n} · {v/n:.1%}',va='center',fontsize=9.5,color='#385263')
        ax.barh(np.arange(4)+(j-1)*.23,values,height=.20,label=label,color=color)
    ax.set_yticks(range(4),list(NAMES.values()));ax.invert_yaxis();ax.set_xlim(0,88)
    ax.set_xticks([0,20,40,60,80]);ax.set_xlabel('각 단절 유형에서 해당 이력이 있었던 비율(%)');ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    fig.legend(loc='lower center',bbox_to_anchor=(.5,.065),frameon=False,ncol=1,fontsize=10.5)
    fig.text(.5,.025,'큰 급락이 보였다는 사실과, 연결을 유지하던 때의 예측 가능한 전조는 구분해야 한다. 초기 이력 부족 17건은 이 그래프에서 제외했다.',ha='center',fontsize=10,color='#526575')
    path=OUT/'01_유형별_앞선급락.png';fig.savefig(path,dpi=165);plt.close(fig);return path


def risk_figure(s,f):
    fig,axes=plt.subplots(2,2,figsize=(14.6,10.1))
    fig.subplots_adjust(left=.09,right=.965,top=.84,bottom=.13,hspace=.61,wspace=.30)
    fig.text(.055,.95,'연결이 유지된 급락을 관찰한 뒤, 무엇이 달랐나',fontsize=21,fontweight='bold')
    fig.text(.055,.895,'새 예측기의 성능이 아닌 기록의 연관성 · 아래 두 그래프는 직전 30초가 연속 연결된 급락만 사용',fontsize=11.5,color='#526575')
    ax=axes[0,0];n=s['pulse_groups']['all']['n'];out=s['pulse_groups']['all']['outcomes']
    order=['no_onset_30','late_under3','early_3to7','long_7to30']
    labels=['30초 내\n새 단절 없음','3초 미만 뒤 단절','3~7초 뒤 단절','7~30초 뒤 단절']
    vals=[out[k]/n*100 for k in order]
    ax.barh(range(4),vals,color=['#6c8c94','#b77052','#218c78','#4389a5'],height=.57)
    for i,(v,k) in enumerate(zip(vals,order)):ax.text(v+1,i,f'{out[k]:,}건 · {v:.1f}%',va='center',fontsize=10)
    ax.set_yticks(range(4),labels,fontsize=10);ax.invert_yaxis();ax.set_xlim(0,111);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('연결 유지 급락 3,830회 중 비율(%)');ax.set_title('급락 후 30초의 결과',loc='left',fontsize=13,pad=17)
    ax=axes[0,1];c=f['comparisons']['continuous_up30']['recent_drop30']['next30']
    vals=[c['risk_without_history']*100,c['risk_with_history']*100]
    ax.bar(range(2),vals,color=['#93a8b2','#168472'],width=.52)
    for i,v in enumerate(vals):ax.text(i,v+.2,f'{v:.2f}%',ha='center',fontsize=14,fontweight='bold')
    ax.set_xticks(range(2),['최근 급락 이력 없음','최근 급락 이력 있음']);ax.set_ylim(0,10);ax.set_ylabel('이후 30초 첫 새 단절 비율(%)')
    ax.set_title('같은 지도·현재 여유 1dB 구간을 맞춘 대조',loc='left',fontsize=12,pad=17)
    ax.text(.5,.90,f'약 {c["risk_ratio"]:.2f}배 · 공간 그룹 95% 구간\n[{c["risk_ratio_ci95"][0]:.2f}, {c["risk_ratio_ci95"][1]:.2f}]',transform=ax.transAxes,ha='center',va='top',fontsize=10.5,color='#3b5b6c')
    ax=axes[1,0];bands=s['pulse_groups']['no_prior_down30']['bands']
    vals=[bands[k]['risk']*100 for k in ('0to3','3to10','10to20','20plus')]
    labels=['0~3dB','3~10dB','10~20dB','20dB 이상']
    ax.bar(range(4),vals,color=['#b77052','#b08b3c','#4b9b8c','#6b8e9f'],width=.60)
    for i,(k,v) in enumerate(zip(('0to3','3to10','10to20','20plus'),vals)):
        row=bands[k];ax.text(i,v+1.2,f'{v:.1f}%\n{row["onset_next30"]}/{row["n"]}',ha='center',fontsize=10)
    ax.set_xticks(range(4),labels);ax.set_ylim(0,68);ax.set_ylabel('급락 후 30초 새 단절 비율(%)');ax.set_title('급락 뒤 남은 통신 여유',loc='left',fontsize=13,pad=17)
    ax=axes[1,1];b=f['backup_transitions']['clean30'];rows=[b['still_redundant'],b['newly_lost_backup'],b['already_vulnerable']]
    vals=[r['onset_next30']/r['n']*100 for r in rows]
    ax.bar(range(3),vals,color=['#4c9487','#b77052','#ac8b49'],width=.60)
    for i,(v,r) in enumerate(zip(vals,rows)):ax.text(i,v+1.2,f'{v:.1f}%\n{r["onset_next30"]}/{r["n"]}',ha='center',fontsize=10)
    ax.set_xticks(range(3),['우회 유지','급락 때 우회 상실','이미 우회 부족']);ax.set_ylim(0,68);ax.set_ylabel('급락 후 30초 새 단절 비율(%)');ax.set_title('단일 링크 상실을 견딜 우회 경로',loc='left',fontsize=12.5,pad=17)
    for ax in axes.ravel():ax.grid(axis='x' if ax is axes[0,0] else 'y',alpha=.10);ax.set_axisbelow(True)
    fig.text(.5,.035,'오른쪽 아래는 현재 여유까지 맞춘 비교가 아니다. 현재 여유·경로·기존 제어의 영향이 함께 섞여 있으며, 재배치의 인과효과를 뜻하지 않는다.',ha='center',fontsize=10,color='#526575')
    path=OUT/'02_급락후_위험과우회.png';fig.savefig(path,dpi=165);plt.close(fig);return path


def choose_examples(pulses):
    candidates=[p for p in pulses if p['previous30_all_connected'] and 3<=p['after_db']<10]
    groups={
        'recovered':[p for p in candidates if p['outcome']=='no_onset_30' and p['recovery_start_s'] is not None and p['recovery_start_s']<=10],
        'lower_stable':[p for p in candidates if p['outcome']=='no_onset_30' and p['recovery_start_s'] is None],
        'future_cut':[p for p in candidates if p['first_onset_lead_s'] is not None and p['first_onset_lead_s']>=3 and (p['recovery_start_s'] is None or p['recovery_start_s']>p['first_onset_lead_s'])]}
    selected={}
    for name,rows in groups.items():
        ordered=sorted(rows,key=lambda p:abs(p['after_db']-7.)+abs(p['drop_db']-20.)*.05)
        for p in ordered:
            m,up,_=trace(p['region']);t=round(p['time_s']*2)
            if t<60 or t+61>=len(m):continue
            if name=='lower_stable' and (m[t:t+61].min()<3 or np.ptp(m[t:t+61])>7 or not up[t:t+61].all()):continue
            selected[name]=p;break
        if name not in selected:raise ValueError('No honest example for '+name)
    return selected


def example_figure(examples):
    fig,axes=plt.subplots(3,1,figsize=(13.2,11.7))
    fig.subplots_adjust(left=.08,right=.97,top=.86,bottom=.10,hspace=.54)
    fig.text(.06,.96,'같은 급락도 이후 전개가 다르다',fontsize=22,fontweight='bold')
    fig.text(.06,.914,'실제 원본 기록의 0.5초 값 · 0초는 연결을 유지한 급락 시각 · 설명을 위해 사후 선정한 사례',fontsize=11.5,color='#526575')
    texts={'recovered':('여유 회복 · 30초 내 새 지속 단절 없음','#168472'),
           'lower_stable':('낮아진 수준에서 안정 · 30초 내 새 지속 단절 없음','#528b9b'),
           'future_cut':('여유 미회복 뒤 새 지속 단절 발생','#b66b48')}
    for ax,(name,p) in zip(axes,examples.items()):
        m,up,_=trace(p['region']);t=round(p['time_s']*2);x=np.arange(-60,61)*.5;y=m[t-60:t+61]
        label,color=texts[name]
        padded=np.r_[True,up[t-60:t+61],True];bounds=np.flatnonzero(padded[1:]!=padded[:-1])
        for a,b in zip(bounds[::2],bounds[1::2]):
            if b-a>=2:ax.axvspan((a-60)*.5,(b-60)*.5,color='#c5765f',alpha=.14)
        ax.plot(x,y,'o-',lw=1.8,markersize=2.4,color=color)
        ax.axvline(0,color='#516d7a',ls='--',lw=1);ax.axhline(0,color='#777',ls=':',lw=1)
        ax.scatter([-.5,0],[m[t-1],m[t]],s=36,color=color,zorder=4)
        ax.set_xlim(-30.5,30.5);ax.set_ylim(-22,45);ax.set_xticks([-30,-20,-10,0,10,20,30]);ax.set_xlabel('급락 시각 기준 시간(초)');ax.set_ylabel('수색망 통신 여유(dB)');ax.grid(alpha=.10)
        ax.set_title(label,loc='left',fontsize=12.5,fontweight='bold',pad=23,color=color)
        detail=f'{p["region"]} · {p["kind"]} · 비행 {p["time_s"]:g}초 · {p["before_db"]:.1f}→{p["after_db"]:.1f}dB'
        ax.text(0,1.025,detail,transform=ax.transAxes,fontsize=9.5,color='#526575')
    fig.text(.5,.035,'붉은 음영은 실제 지속 단절이다. 기존 V7 재배치가 작동하던 비행이므로 회복을 무개입 자연 회복으로 단정하지 않는다.',ha='center',fontsize=10,color='#526575')
    path=OUT/'03_급락후_실제전개.png';fig.savefig(path,dpi=165);plt.close(fig);return path


def original_examples():
    examples=read(ROOT/'output/outage_patterns_20261105/evidence.json')['examples'];rows=[]
    for shape,p in examples.items():
        m,up,pulse=trace(p['region']);onset=round(p['onset_s']*2)
        times=np.arange(max(1,onset-60),onset-5)
        drop=m[times-1]-m[times]
        items=[]
        for t,d in zip(times,drop):
            if d<6.:continue
            items.append(dict(relative_s=(int(t)-onset)*.5,drop_db=round(float(d),3),
                         before_db=round(float(m[t-1]),3),after_db=round(float(m[t]),3),
                         network_before=bool(up[t-1]),network_after=bool(up[t])))
        rows.append(dict(pattern=shape,region=p['region'],onset_s=p['onset_s'],prior_jumps=items))
    return rows


def report(s,f,examples,old,figures):
    lines=['# Final V1 · 단절되지 않은 급락의 선행 가치 분석','',
      '최근 30초에 연결을 유지한 급락이 있었는지, 그 기록이 추가 단절 위험을 알려 주는지 조사했다. '
      f'원본 V7 비행 345개·{s["hours"]:.2f}시간·{s["radio_frames"]:,}개 0.5초 프레임을 복원했다. '
      '모든 링크 판정, 기존 2,572개 새 지속 단절과 기존 6초 여유 값이 원본과 일치했다. 정상 구간도 포함했다.','',
      '결과: 급락 이력에는 약한 추가 정보가 있다. 같은 지도와 현재 여유를 맞춘 뒤에도 위험 차이가 남았다. '
      '다만 연결을 유지한 급락의 84.2%에서는 이후 30초 내 새 지속 단절이 없었다. '
      '이력 하나로 즉시 재배치하는 기준은 부족하다. 남은 여유, 회복, 우회 경로, 앞으로의 이동을 함께 판단하는 방향이 타당하다.','',
      '## 정의와 해석 범위','',
      '- 급락은 직전 0.5초에 전체 수색망 여유가 6dB 이상 감소한 경우다. 연결 유지 급락은 직전·현재 그래프가 연결되고 현재 여유가 0dB 이상인 경우다.',
      '- 여유는 가장 약한 수색기 쌍의 최선 중계 경로가 갖는 병목 SINR 여유다. 0dB는 수신 전력이 0이라는 뜻이 아니다. 실제 단절은 데이터와 ACK의 양방향 홉 성공확률로 별도 판정한다.',
      '- 인접 급락은 과거 기록만 이용해 최소 3초 간격으로 묶었다. 이후 30초와 지속성 판정에 필요한 자료가 온전히 있는 3,830회를 분석했다. 끝부분의 후속 이력 부족 급락은 별도 집계했다.',
      '- 전체망 전파 정답을 복원한 진단이다. 실제 첫 중계기에 도착한 관측이나 학습한 예측기의 입력 성능이 아니다. 원래 정책이 이미 중계기를 움직인 기록이므로, 이후 단절이 없다는 사실만으로 불필요한 경보나 자연 회복이라고 단정할 수 없다.','',
      '## 유형별 단절 3~30초 전의 기록','',
      '30초 이력이 완전한 2,555건만 같은 분모로 비교했다. 마지막 실제 단절 순간의 급락은 제외했다.','',
      '| 유형 | 사건 수 | 앞선 급락 전체 | 급락 전후 연결 유지 | 그 급락부터 이번 단절 전까지 계속 연결 |',
      '|---|---:|---:|---:|---:|']
    for k,name in NAMES.items():
        row=s['types'][k];n=row['n30'];a=row['complete30']
        lines.append('| '+name+f' | {n:,} | '+' | '.join(f'{a[key]:,} ({a[key]/n:.1%})' for key in ('any_prior_jump','still_connected_jump','uninterrupted_to_onset_jump'))+' |')
    lines+=['','전체 2,572건 기준으로 연결 유지 급락은 540건, 앞선 별도 단절 없이 현재 사건까지 연결이 이어진 급락은 294건(11.4%)에서 있었다. '
       '모든 유형에서 이런 이력이 나타날 수 있지만 모든 단절이 이 전조를 갖는 것은 아니다. 앞선 큰 하락 중에는 이미 발생했던 별도 단절도 섞여 있다.','',
       '## 연결 유지 급락 후 30초','',
       '| 결과 | 횟수 | 비율 |','|---|---:|---:|']
    labels={'no_onset_30':'새 지속 단절 없음','late_under3':'3초 미만 뒤 첫 단절','early_3to7':'3초 이상 7초 미만 뒤 첫 단절','long_7to30':'7초 이상 30초 이내 첫 단절'}
    for key,label in labels.items():
        n=s['pulse_groups']['all']['outcomes'][key];lines.append(f'| {label} | {n:,} | {n/3830:.2%} |')
    lines+=['', '3초 이상 앞선 급락은 444회였다. 여러 급락이 같은 단절과 연결될 수 있으므로 444를 고유 탐지 사건 수나 재현율로 사용할 수 없다. '
       '단절이 없던 3,225회 중 2,171회는 급락 전보다 3dB 낮은 수준까지라도 2초 이상 회복했다. '
       '나머지 1,054회는 이 회복 기준을 충족하지 않았어도 30초 내 새 지속 단절이 없었다. '
       '낮아진 수준 자체가 안정된 상태일 수 있으므로, 급락 전 수준을 무조건 되찾도록 움직이는 목적도 검증이 필요하다.','',
       '## 현재 여유가 같아도 급락 이력에 정보가 있는가','',
       '같은 지도·현재 여유 1dB 구간에서 급락 이력 있음/없음을 비교했다. 각 집단이 20프레임 이상 있는 구간만 사용했고 '
       '두 집단 표본 수 중 작은 수로 같은 가중을 적용했다. 공간 그룹 전체를 5,000번 재표집했다. 중첩된 30초 창을 독립 사건으로 취급하지 않았다.','',
       '| 비교 범위 | 급락 이력 없음 | 급락 이력 있음 | 위험비 | 공간 그룹 95% 구간 |','|---|---:|---:|---:|---:|']
    for scope,label in [('currently_up','현재 연결'),('continuous_up30','직전 30초 내내 연결')]:
        v=f['comparisons'][scope]['recent_drop30']['next30'];lo,hi=v['risk_ratio_ci95']
        lines.append(f'| {label} | {v["risk_without_history"]:.2%} | {v["risk_with_history"]:.2%} | {v["risk_ratio"]:.2f}배 | [{lo:.2f}, {hi:.2f}] |')
    e=f['comparisons']['continuous_up30']['recent_drop30']['early3to30'];lo,hi=e['risk_ratio_ci95']
    lines+=['',f'첫 단절이 3~30초 뒤인 경우만 보면 직전 30초 연속 연결 시점에서 {e["risk_without_history"]:.2%}→{e["risk_with_history"]:.2%}, '
       f'위험비 {e["risk_ratio"]:.2f}배, 95% [{lo:.2f},{hi:.2f}]였다. '
       '이는 기록의 조건부 연관성이다. 모델 정밀도·보정 확률·신규 자료의 예측 성능 향상률과 구분한다. '
       '미회복 여부를 더한 별도 대조의 3초 이상 결과는 불확실성 구간이 위험비 1을 포함했다. 미회복 지표의 추가 효과를 확정할 수 없다.','',
       '## 남은 여유와 우회 경로','',
       '이 표는 직전 30초에 실제 연결이 한 번도 끊기지 않은 급락 3,315회다.','',
       '| 급락 후 여유 | 급락 횟수 | 이후 30초 첫 단절 | 비율 |','|---|---:|---:|---:|']
    for key,label in [('0to3','0 이상 3dB 미만'),('3to10','3 이상 10dB 미만'),('10to20','10 이상 20dB 미만'),('20plus','20dB 이상')]:
        v=s['pulse_groups']['no_prior_down30']['bands'][key];lines.append(f'| {label} | {v["n"]:,} | {v["onset_next30"]:,} | {v["risk"]:.2%} |')
    lines+=['','| 우회 경로 변화 | 급락 횟수 | 이후 30초 첫 단절 | 비율 |','|---|---:|---:|---:|']
    for key,label in [('still_redundant','급락 전후 단일 링크 상실에 견딜 우회 유지'),('newly_lost_backup','급락 때 해당 우회 상실'),('already_vulnerable','급락 전부터 해당 우회 부족')]:
        v=f['backup_transitions']['clean30'][key];lines.append(f'| {label} | {v["n"]:,} | {v["onset_next30"]:,} | {v["onset_next30"]/v["n"]:.2%} |')
    lines+=['','우회 표는 현재 여유까지 맞춘 대조가 아니므로 독립적인 5배 예측 효과로 해석하지 않는다. '
       '우회 상실 152회 중 3초 이상 뒤 단절은 37회였다. 나머지 단절 41회는 3초 미만 뒤에 왔다. '
       '실제 우회가 사라진 것을 본 다음에만 움직이면 늦을 수 있다. 예정 경로에서 우회가 사라질 시점을 먼저 계산하는 것이 필요하다.','',
       '## 급락 원인 항과 전송 품질','',
       '직전 항으로 되돌렸을 때 급락의 절반 이상이 회복되는지를 계산했다. 같은 급락이 여러 항에 해당할 수 있다. '
       '건물의 공중 링크 추가 손실 735/3,830회(19.2%), 간섭 수신항 변화 3,098/3,830회(80.9%)였다. '
       '간섭 수신항 중 간헐 간섭 시작 전환은 55회, 고정 간섭원에 대한 이동 중 수신 경로 변화는 3,043회였다. '
       '단절 순간에서 건물 차폐가 우세했던 이전 조사와 대상이 다르다. 여기서는 연결을 유지한 급락만 조사했다. '
       '간섭원의 실제 위치·향후 전환 시각은 원인 분석 전용이며 예측 입력으로 사용할 수 없다.','',
       '망 단절이 없던 3,225회 중 117회에는 다음 30초 안에 적시 전달률의 1초 창이 90% 미만인 시점이 있었다. '
       '급락 시점에는 서비스가 정상이었지만 이후 이 기준을 넘지 못한 경우는 70회다. '
       '이는 실제 망 연결과 데이터 기한 준수가 별개임을 보여 주며, 임무 전체 실사용 신뢰도를 뜻하지 않는다.','',
       '## 앞서 본 네 대표 그래프의 급락 정체','']
    for row in old:
        if row['prior_jumps']:
            text='; '.join(f'{q["relative_s"]:g}초에 {q["before_db"]:.1f}→{q["after_db"]:.1f}dB, '+('전후 연결 유지' if q['network_before'] and q['network_after'] else '연결이 끊긴 상태로 전환/지속') for q in row['prior_jumps'])
        else:text='단절 3~30초 전에는 0.5초 6dB 이상 급락 없음'
        lines.append(f'- {NAMES[row["pattern"]]} 대표 {row["region"]}: {text}.')
    lines+=['','## 다음 알고리즘 검증에 반영할 방향','',
       '1. 급락을 즉시 이동 명령 대신 30초 위험 이력으로 보관한다. 발생 시각·크기·현재 여유·회복 추세·관측 신선도를 함께 기록한다.',
       '2. 연결을 유지하는 정상 급락과, 그 뒤 실제 단절이 온 급락을 모두 학습 대조에 넣는다. 현재값만 추가한 모델과 30초 이력을 추가한 모델을 비교한다.',
       '3. 7~30초 위험은 주의 상태와 후보 경로 준비에 사용한다. 실제 이동 여부는 더 가까운 시점의 여유·추정 우회·예정 경로와 이동 완료 시간을 함께 비교한다.',
       '4. 최근 급락과 우회 부족이 겹치면 예정 경로의 우회 소실을 먼저 예측한다. 후보 이동이 수색망 최소 여유와 단일 링크 우회를 함께 개선하는지 확인한다. 급락 전 수치 복구만 목표로 삼지 않는다.',
       '5. 우선 기존 1Hz 수신을 쓰고 0.5초 판단 환경에서 시험한다. 10Hz 전 기체 비컨은 이전 시험에서 전송 비용이 컸다. 새 보고가 필요한 링크만 제한적으로 확인하는 안은 별도 대조한다.',
       '6. 실제 수신된 과거 자료만으로 검증한다. 245개 훈련 지도와 보정·선정·새 확인을 분리하고, 같은 경보 비용에서 3초 이상 탐지와 실제 단절 시작률을 각각 평가한다.',
       '', '이번 작업은 기록 분석이다. 새로운 예측기 학습과 재배치 효과 측정은 수행하지 않았다. 기존 Final V1 제어·모델은 보존했다. '
       '345개 지도에는 이미 진단에 공개한 이전 최종 지도도 포함되어 있으므로 이 자료로 새 미공개 지도 성능을 주장할 수 없다.','']
    for p in figures:lines+=['!['+p.stem+']('+p.name+')','']
    path=OUT/'급락이력_선행가치_분석.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');return path


def web(s,f,figures):
    source=read(ROOT/'output/outage_patterns_20261105/evidence.json')
    trows=[]
    for k,name in NAMES.items():
        row=s['types'][k];n=row['n30'];c=row['complete30']
        trows.append('<tr><td>'+name+'</td><td>'+str(n)+'</td>'+''.join(f'<td>{c[key]} · {c[key]/n:.1%}</td>' for key in ('any_prior_jump','still_connected_jump','uninterrupted_to_onset_jump'))+'</tr>')
    brows=[]
    for key,label in [('0to3','0~3dB'),('3to10','3~10dB'),('10to20','10~20dB'),('20plus','20dB 이상')]:
        v=s['pulse_groups']['no_prior_down30']['bands'][key]
        brows.append(f'<tr><td>{label}</td><td>{v["n"]:,}</td><td>{v["onset_next30"]:,}</td><td>{v["risk"]:.2%}</td></tr>')
    r=f['comparisons']['continuous_up30']['recent_drop30']['next30']
    template=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final V1 · 급락 이력의 선행 가치</title>
    <style>:root{color-scheme:light;--ink:#183344;--muted:#567080;--line:#d6e1e6}*{box-sizing:border-box}body{margin:0;background:#f0f5f6;color:var(--ink);font:15px/1.75 "Malgun Gothic",-apple-system,system-ui,sans-serif}main{max-width:1170px;margin:auto;padding:24px}header{padding:30px 34px;border-radius:18px;background:#153143;color:#f6fafb}header small{color:#8ed9ca;letter-spacing:.1em;font-weight:700}h1{font-size:clamp(26px,4vw,38px);line-height:1.3;margin:10px 0 14px}header p{color:#d2e3eb;margin:8px 0}.nav{display:flex;flex-wrap:wrap;gap:18px;margin-top:18px}a{color:#146b9f;text-underline-offset:4px}header a{color:#a8e5d9}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.stat,.card{background:white;border:1px solid var(--line);border-radius:13px}.stat{padding:15px 18px}.stat strong{display:block;font-size:26px}.stat span,.note{font-size:13px;color:var(--muted)}.card{padding:24px;margin-top:16px}h2{font-size:21px;margin:0 0 10px}p{margin:10px 0}.callout{padding:13px 17px;background:#eaf5f0;border-left:4px solid #168472;border-radius:5px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:13px}th,td{padding:11px 10px;border-bottom:1px solid #dfe7eb;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#f1f6f7}img{display:block;width:100%;height:auto;border:1px solid #e1e9ed;border-radius:10px;margin:15px 0}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.twocol .card{margin-top:0}summary{cursor:pointer;font-weight:700}footer{padding:24px 4px;color:var(--muted);font-size:12px}button{border:0;border-radius:8px;padding:10px 15px;background:#168472;color:white;font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #5599b1;outline-offset:3px}@media(max-width:650px){main{padding:12px}header{padding:23px 21px}.card{padding:17px}.stats{grid-template-columns:1fr 1fr;gap:8px}.stat{padding:12px}.twocol{grid-template-columns:1fr}h2{font-size:19px}}</style></head><body><main>
    <header><small>FINAL V1 · PRECURSOR RECORD ANALYSIS</small><h1>끊기지 않은 급락도<br>다음 단절의 신호일까</h1><p><b>급락 이력에는 추가 정보가 있었습니다.</b> 다만 급락 이후 남은 여유와 우회 경로를 함께 봐야 합니다.</p><p>기존 V7 원본 비행의 정상 구간까지 복원한 기록 분석입니다. 새 예측기의 정확도나 재배치 개선률을 측정한 결과와 구분합니다.</p><div class="nav"><a href="./">비행 재생</a><a href="history.html">30초 이력 조회</a><a href="fast-control.html">빠른 제어 실험</a></div></header>
    <div class="stats"><div class="stat"><strong>345개</strong><span>원본 비행 지도</span></div><div class="stat"><strong>182.9시간</strong><span>정상·단절 구간 전체 복원</span></div><div class="stat"><strong>3,830회</strong><span>연결을 유지한 급락 묶음</span></div><div class="stat"><strong>1.30배</strong><span>현재 여유를 맞춘 다음 단절 비율</span></div></div>
    <section class="card"><h2>급락 이력은 의미가 있었다</h2><div class="callout"><b>같은 지도·현재 여유 1dB 구간</b>을 맞추고 직전 30초 내내 연결됐던 시점끼리 비교했습니다. 이후 30초 내 첫 단절 비율은 <b>급락 이력 없음 5.35% → 있음 6.95%</b>였습니다. 위험비 1.30배, 공간 그룹 재표집 95% 구간 1.06~1.58배입니다.</div><p>첫 단절이 3초 이상 30초 이내인 경우만 보면 5.01→6.41%였습니다. 급락 이력이 현재 신호 수준 이상의 맥락을 담을 가능성을 보여 줍니다.</p><p class="note">이는 기록의 조건부 연관성입니다. 6.95%를 예측기 정밀도나 보정 확률로 해석하지 않습니다. 과거 30초에 0.5초당 6dB 이상 급락이 있었는지를 사용했습니다.</p></section>
    <section class="card"><h2>모든 유형에 나타날 수 있지만, 전조의 범위는 달랐다</h2><p>단절 3~30초 전의 급락을 조사했습니다. 앞서 이미 끊겼던 구간도 30초 그래프에 들어올 수 있어 별도로 나눴습니다.</p><div class="table-wrap"><table><thead><tr><th>유형</th><th>30초 이력 사건</th><th>앞선 급락 전체</th><th>전후 연결 유지</th><th>이번 단절까지 계속 연결</th></tr></thead><tbody>__TYPE_ROWS__</tbody></table></div><p class="note">마지막 실제 단절 순간의 급락은 제외했습니다. 전체 새 단절 2,572건 중 30초 이력이 충분한 2,555건을 표에 사용했습니다. 전체 사건에서 계속 연결된 같은 구간의 급락은 294건(11.4%)에 있었습니다.</p><img src="__FIG1__" alt="단절 유형별 앞선 급락과 연결을 유지한 급락의 비율 비교"></section>
    <section class="card"><h2>급락만으로 곧바로 이동하면 생기는 문제</h2><p>연결 유지 급락 3,830회 중 <b>3,225회(84.2%)는 이후 30초 안에 새 지속 단절이 없었습니다.</b> 3초 미만 뒤 단절 161회, 3~7초 뒤 112회, 7~30초 뒤 332회였습니다.</p><p>단절이 없던 급락 중 2,171회는 앞선 수준에 가깝게 회복했고, 1,054회는 회복 기준을 충족하지 않아도 30초 내 단절이 없었습니다. 낮아진 수준에서 안정될 수도 있어 급락 전 수치를 무조건 회복시키는 목적은 과도한 이동을 만들 수 있습니다.</p><p class="note">회복은 급락 전보다 3dB 낮은 수준 이상에서 2초간 유지한 경우입니다. 기존 정책이 이미 움직이던 비행이므로, 단절이 없던 사례를 모두 자연 회복이나 불필요한 경보로 부를 수는 없습니다.</p><img src="__FIG3__" alt="연결 유지 급락 뒤 회복·낮은 수준 안정·새 단절로 나뉜 실제 사례"></section>
    <section class="card"><h2>남은 여유와 우회 경로가 중요했다</h2><p>아래 표는 급락 전 30초 내내 연결됐던 3,315회만 사용했습니다.</p><div class="table-wrap"><table><thead><tr><th>급락 후 여유</th><th>급락 횟수</th><th>30초 내 첫 단절</th><th>비율</th></tr></thead><tbody>__BAND_ROWS__</tbody></table></div><p>단일 링크 상실을 견딜 우회가 유지된 경우는 <b>305/3,073회·9.9%</b>, 급락 때 그 우회가 없어진 경우는 <b>78/152회·51.3%</b>에서 새 단절이 왔습니다.</p><p class="note">우회 비교는 현재 여유까지 맞춘 대조가 아닙니다. 우회 상실의 독립적인 5배 효과를 측정한 값으로 해석하지 않습니다. 우회 상실 후 단절 78회 중 41회는 3초 미만 뒤에 왔으므로, 사라진 다음에만 대응하면 늦을 수 있습니다.</p><img src="__FIG2__" alt="급락 후 단절 비율, 현재 여유를 맞춘 위험 차이와 우회 경로 대조"></section>
    <div class="twocol"><section class="card"><h2>이 급락의 원인은 무엇이었나</h2><p>직전 원인 항으로 되돌렸을 때 급락의 절반 이상이 회복된 경우는 간섭 수신항 <b>3,098회·80.9%</b>, 공중 링크의 건물 추가 손실 <b>735회·19.2%</b>였습니다. 항들은 겹칠 수 있습니다.</p><p>간섭 수신항 변화 중 55회는 간헐 간섭 시작 전환, 3,043회는 이동에 따른 고정 간섭원 수신 경로 변화였습니다. 실제 단절 순간의 차폐 중심 분석과 대상이 다릅니다.</p><p class="note">간섭원의 실제 위치·향후 전환 시각은 원인 조사 전용입니다. 온라인 예측에 이 정답을 넣을 수 없습니다.</p></section><section class="card"><h2>예측과 재배치에 반영할 방향</h2><p><b>30초 위험 이력</b>에 급락 시각·크기·회복 추세·관측 신선도를 보관합니다.</p><p><b>주의 상태</b>에서는 후보 경로를 준비하고, <b>남은 여유·추정 우회·예정 경로</b>가 함께 나빠질 때 이동을 검토합니다.</p><p><b>우회가 없어질 예정 시각</b>을 실제 이동 완료 시간보다 먼저 예측해야 합니다. 단절 후 복구보다 앞선 예방 판단에 사용합니다.</p><p class="note">이 방향의 새 학습과 실제 예방 효과는 아직 검증하지 않았습니다. 기존 1Hz 수신과 0.5초 판단부터 대조하는 것이 다음 단계입니다.</p></section></div>
    <details class="card"><summary>근거와 한계</summary><p>1,316,892개 0.5초 전파 프레임의 링크 판정과 기존 2,572개 새 지속 단절·6초 여유 값이 원본과 일치했습니다. 직접·중계 경로로 전체 수색망이 1초 이상 연결되지 않으면 지속 단절입니다.</p><p>여유는 가장 약한 수색기 쌍의 최선 중계 경로에 대한 SINR 기준의 근사 여유입니다. 0dB는 신호가 사라진다는 뜻이 아닙니다. 전체망 전파 정답으로 복원한 진단이며 첫 중계기가 실제 수신할 수 있었던 자료와 구분합니다.</p><p>정상 구간도 포함했습니다. 현재 연결된 약 124만 시점과 직전 30초 연속 연결된 약 115만 시점을 조사했습니다. 같은 지도·현재 여유 1dB 구간의 공통 표본 가중으로 대조하고 공간 그룹을 단위로 재표집했습니다.</p><p>기존 V7 재배치가 작동한 기록의 연관성이므로 재배치의 인과효과를 추정한 것이 아닙니다. 이번 자료로 새 모델의 AUROC·정밀도·실운용 오경보율·미공개 지도 성능을 주장하지 않습니다. 학습·제어와 가중치는 바꾸지 않았습니다.</p></details><footer>Final V1 · 2026-11-05 · 기록 분석 완료 · 본인 계정 전용 요약</footer></main></body></html>'''
    page=template.replace('__TYPE_ROWS__',''.join(trows)).replace('__BAND_ROWS__',''.join(brows))
    for key,p in zip(('__FIG1__','__FIG2__','__FIG3__'),figures):page=page.replace(key,img(p))
    for forbidden in ('C:\\Users','C:/Users','archive_path','Bearer ','source_repository_credential'):
        assert forbidden not in page,forbidden
    path=OUT/'precursor.html';path.write_text(page,encoding='utf-8');return path


def main():
    s=read(OUT/'summary.json');f=read(OUT/'fine_controls.json');p=read(OUT/'pulse_records.json')
    examples=choose_examples(p);old=original_examples()
    figures=[type_figure(s),risk_figure(s,f),example_figure(examples)]
    note=report(s,f,examples,old,figures);page=web(s,f,figures)
    save(OUT/'report_evidence.json',dict(source_summary_sha256=digest(OUT/'summary.json'),
        fine_control_sha256=digest(OUT/'fine_controls.json'),selected_examples=examples,
        original_examples=old,figures={p.name:digest(p) for p in figures},
        report_sha256=digest(note),page_sha256=digest(page),page_bytes=page.stat().st_size,
        builder_sha256=digest(__file__)))
    print(json.dumps(dict(page=str(page),page_bytes=page.stat().st_size,page_sha256=digest(page),old_examples=old),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
