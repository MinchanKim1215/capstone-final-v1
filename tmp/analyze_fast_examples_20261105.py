from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final_v1_fast.protocol import read,write,sha,ARTIFACTS

def main():
    counts=dict(fewer=0,more=0,same=0)
    examples=[]
    for path in sorted((ARTIFACTS/'confirm/p05_b1').glob('*.json')):
        fast=read(path);base=read(ARTIFACTS/'confirm/p3_b1'/path.name)
        counts['fewer' if fast['onsets']<base['onsets'] else 'more' if fast['onsets']>base['onsets'] else 'same']+=1
        with np.load(fast['archive_path'],allow_pickle=False) as f,np.load(base['archive_path'],allow_pickle=False) as b:
            difference=np.flatnonzero((np.abs(f['positions']-b['positions'])>1e-5).any(axis=(1,2)))
            if not len(difference):continue
            position_frame=int(difference[0])
            np.testing.assert_array_equal(f['positions'][:position_frame],b['positions'][:position_frame])
            np.testing.assert_array_equal(f['links'][:position_frame],b['links'][:position_frame])
            for event in fast['interventions']:
                t=event['time_s']
                if not event['actual_connected_at_request'] or t+12>24:continue
                match=[(a,z) for a,z in base['cuts'] if 3.<=a-t<7.]
                if not match:continue
                a,z=match[0]
                if not f['network'][int(t*10):int((t+12)*10)].all():continue
                r=event['changed_relay']-fast['search_drones']
                changed=np.flatnonzero((f['applied'][:,r]==event['new_offset']).all(axis=1))
                if not len(changed):continue
                applied=float(changed[0]*.1)
                examples.append(dict(case_key=fast['case_key'],spec=fast['spec'],start_s=fast['start_s'],
                     calculation_start_s=t,result_ready_s=event['result_ready_s'],
                     coordinator_earliest_apply_s=event['earliest_apply_s'],
                     changed_relay=event['changed_relay'],actual_offset_received_s=applied,
                     first_position_difference_s=position_frame*.1,baseline_cut_start_s=a,
                     baseline_cut_end_s=z,connected_after_request_s=12.,same_positions_and_links_before_s=position_frame*.1,
                     selection='Post-hoc illustrative case, not additional validation or a prevention-rate estimate'))
                break
    write(ARTIFACTS/'paired_examples.json',dict(branch_outcomes=counts,examples=examples))
    print(json.dumps(dict(counts=counts,examples=examples),ensure_ascii=False,indent=2))
    if not examples:return
    e=examples[0]
    from matplotlib.font_manager import FontProperties
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fast=read(ARTIFACTS/'confirm/p05_b1'/(e['case_key']+'.json'))
    base=read(ARTIFACTS/'confirm/p3_b1'/(e['case_key']+'.json'))
    fig,(a,b)=plt.subplots(2,1,figsize=(13.5,8.3),gridspec_kw={'height_ratios':[2,1]})
    fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.13,hspace=.5)
    fig.text(.06,.95,'빠른 판단이 실제 이동으로 이어진 사례',fontsize=22,fontweight='bold')
    fig.text(.06,.89,f'{e["spec"]["region"]} · {e["spec"]["kind"]} · 원본 {e["start_s"]:g}초 위치에서 새로 시작한 24초 시험',fontsize=11,color='#526575')
    for row,label,color in [(base,'3초 판단','#b76c51'),(fast,'0.5초 판단','#168472')]:
        with np.load(row['archive_path']) as z:a.plot(np.arange(240)*.1,z['margin_db'],color=color,lw=2,label=label)
    for start,stop in base['cuts']:a.axvspan(start,stop,color='#c4654e',alpha=.12)
    a.axhline(0,color='#777',ls=':',lw=1);a.set_xlim(0,24);a.set_ylabel('수색망 통신 여유(dB)');a.set_xlabel('새 시험 시작 후 시간(초)');a.grid(alpha=.12);a.legend(frameon=False)
    points=[('위험 판단',e['calculation_start_s']),('계산 완료',e['result_ready_s']),('명령 수신',e['actual_offset_received_s']),('실제 경로 차이',e['first_position_difference_s'])]
    for i,(label,t) in enumerate(points):
        b.scatter(t,i,s=75,color='#168472');b.text(t+.17,i,f'{t:.1f}초',va='center',fontsize=10)
    b.axvline(e['baseline_cut_start_s'],color='#b76c51',ls='--',label=f'3초 판단의 단절 시작 {e["baseline_cut_start_s"]:.1f}초')
    b.set_yticks(np.arange(4),[q[0] for q in points]);b.invert_yaxis();b.set_xlim(0,24);b.set_xlabel('새 시험 시작 후 시간(초)');b.grid(axis='x',alpha=.12);b.legend(frameon=False,loc='lower right')
    fig.text(.5,.04,'사후 선정한 설명용 사례다. 0.1초 값은 새 시뮬레이션이며 실제 기체 측정이 아니다. 최초 상태·앞선 링크의 일치를 검증했다.',ha='center',fontsize=10,color='#526575')
    fig.savefig(ROOT/'final_v1_fast/빠른제어_예방사례.png',dpi=165);plt.close(fig)

if __name__=='__main__':main()
