"""Produce plots, an evidence note, and an offline inspectable history viewer."""
from pathlib import Path
import json
import hashlib
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/outage_history_20261105'
SOURCE=OUT/'long_history.json'
NAMES={'abrupt':'급락형','gradual':'점진형','mixed':'혼합형','other':'기타형','insufficient_history':'이력 부족'}
COLORS={'abrupt':'#b65337','gradual':'#168472','mixed':'#b08021','other':'#62788d'}
ORDER=['abrupt','gradual','mixed','other']
KINDS={'clear':'무간섭','intermittent':'간헐 간섭','continuous':'지속 간섭'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def figures(data):
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(14.5,10.8))
    fig.subplots_adjust(left=.065,right=.975,top=.86,bottom=.13,hspace=.57,wspace=.18)
    fig.text(.055,.965,'같은 사건을 단절 30초 전부터 다시 보기',fontsize=22,fontweight='bold')
    fig.text(.055,.918,'이전 6초 그래프의 대표 사건을 그대로 확장 · 모든 점은 원본과 같은 0.5초 전파 계산',fontsize=12,color='#425e70')
    for ax,kind in zip(axes.ravel(),ORDER):
        e=data['examples'][kind]
        y=np.array(e['margin_db'])
        x=np.arange(-len(y)+1,1)*.5
        for a,b in e['previous_persistent_outages_relative_s']:
            ax.axvspan(a,b,color='#cd7867',alpha=.15)
        ax.axvspan(-6,0,color='#6c9caa',alpha=.08)
        ax.axhline(0,color='#77818b',lw=1,ls=':')
        ax.plot(x,y,'o-',color=COLORS[kind],lw=2.1,markersize=3)
        for lead in (30,20,10,6,.5):
            v=y[-1-int(lead*2)]
            text=f'{v:.1f}'
            ax.annotate(text,(-lead,v),xytext=(3,9),textcoords='offset points',fontsize=9,
                        color=COLORS[kind],fontweight='bold')
        ax.set_xlim(-30.8,.7)
        ax.set_ylim(-23,48)
        ax.set_xticks([-30,-25,-20,-15,-10,-6,0])
        ax.set_xlabel('이번 단절 시작 기준 시간(초)')
        ax.set_ylabel('수색망 통신 여유(dB)')
        ax.grid(alpha=.12)
        ax.set_title(f'{NAMES[kind]} · {e["region"]}',loc='left',fontsize=12.5,fontweight='bold',pad=26,color=COLORS[kind])
        ax.text(0,1.025,f'{KINDS[e["kind"]]} · 시작 {e["onset_s"]:g}초 · 지속 {e["duration_s"]:g}초',
                transform=ax.transAxes,fontsize=9.5,color='#425e70')
    fig.text(.5,.061,'푸른 음영 = 앞서 본 마지막 6초  |  붉은 음영 = 이번 사건 전에 발생했던 별도 지속 단절',ha='center',fontsize=11)
    fig.text(.5,.025,'전체망의 시뮬레이션 전파 여유를 복원한 진단 그래프다. 실제 수신 관측·예측 성능과 구분한다.',ha='center',fontsize=10,color='#526575')
    path=OUT/'01_30초_실제사건.png'
    fig.savefig(path,dpi=165)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14.7,6.8),sharey=True)
    fig.subplots_adjust(left=.07,right=.975,top=.78,bottom=.26,wspace=.12)
    fig.text(.055,.94,'긴 이력에서 보이는 유형별 차이',fontsize=22,fontweight='bold')
    fig.text(.055,.875,'각 선의 대상 사건은 30초 이력이 모두 있는 동일 집합 · 점은 시각별 중앙값',fontsize=12,color='#425e70')
    times=np.arange(-30,.01,.5)
    events=[e for f in data['flights'] for e in f['events']]
    for ax,clean in zip(axes,(False,True)):
        for k in ORDER:
            rows=[e for e in events if e['pattern']==k and e['available_history_s']>=30
                  and (not clean or not e['previous_persistent_outages_relative_s'])]
            median=np.median(np.array([e['margin_db'] for e in rows]),axis=0)
            ax.plot(times,median,lw=2.2,color=COLORS[k],label=f'{NAMES[k]} {len(rows):,}건')
        ax.axhline(0,color='#77818b',lw=1,ls=':')
        ax.axvspan(-6,0,color='#6c9caa',alpha=.08)
        ax.set_xlim(-30.5,.5)
        ax.set_ylim(-18,42)
        ax.set_xlabel('단절 시작 기준 시간(초)')
        ax.set_xticks([-30,-20,-10,-6,0])
        ax.grid(alpha=.12)
        ax.set_title('앞선 단절 포함' if not clean else '앞선 30초에 지속 단절이 없었던 사건만',loc='left',fontsize=12,pad=18)
        ax.legend(loc='lower left',fontsize=10,frameon=False)
    axes[0].set_ylabel('수색망 통신 여유 중앙값(dB)')
    fig.text(.5,.125,'연결선은 여러 사건의 시각별 중앙값이며, 단일 비행 궤적이나 예측 곡선이 아니다.',ha='center',fontsize=10,color='#526575')
    fig.text(.5,.072,'6초로 정의한 기존 분류는 그대로 유지했다. 정상 구간과의 대조 전이므로 이 표로 오경보나 탐지율을 판단할 수 없다.',ha='center',fontsize=10,color='#526575')
    path2=OUT/'02_30초_유형별중앙값.png'
    fig.savefig(path2,dpi=165)
    plt.close(fig)
    return [path,path2]


def report(data, images):
    lines=['# Final V1 · 단절 전 10·20·30초 이력 확장','',
        '345개 지도의 원본 V7 비행을 그대로 사용했다. 새 지속 단절 2,572건 중 10초 이력이 있는 사건은 2,565건, '
        '20초는 2,559건, 30초는 2,555건이다. 초기 비행 구간의 부족한 이력은 채우지 않았다. '
        f'총 {data["verified_radio_frames"]:,}개 전파 프레임의 링크 판정이 원본과 일치했고, 모든 기존 6초 여유 값도 소수 둘째 자리까지 일치했다.','',
        '## 이전에 본 대표 사건의 긴 이력','',
        '| 유형 | 지도 | 30초 전 | 20초 전 | 10초 전 | 6초 전 | 0.5초 전 | 시작 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for k in ORDER:
        e=data['examples'][k]
        vals=[e['margin_db'][-1-int(t*2)] for t in (30,20,10,6,.5,0)]
        lines.append('| '+NAMES[k]+' | '+e['region']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['', '단위는 dB다. 0dB는 신호 전력이 0이라는 뜻이 아니다. 기준 SINR까지 남은 여유가 0이라는 뜻이다. '
        '전체 수색망에서 가장 약한 수색기 쌍의 최선 중계 경로를 기준으로 한다. 실제 지속 단절은 데이터와 ACK의 '
        '양방향 성공확률로 판정하므로, 이 대칭 SINR 여유는 경계의 근사 지표다.','',
        '점진형 대표 사건은 30초 전 34.22dB였고 20초 전 18.01dB로 감소했다. 6초 전부터 낮은 것처럼 보인 것은 '
        '짧은 표시 구간의 영향이었다. 기타형 대표 사건은 30초 전에도 2.54dB였으며, 앞선 30초에 별도 단절 세 구간이 있었다. '
        '따라서 장기간 여유 부족과 한 번의 점진 감소를 같은 규칙으로 해석하면 안 된다.','',
        '## 유형 전체의 중앙값','',
        '30초 이력이 모두 있는 같은 사건 집합을 시간마다 유지했다. 기존 마지막 6초의 분류를 바꾸지 않았다.','',
        '| 유형 | 사건 수 | 30초 전 | 20초 전 | 10초 전 | 6초 전 | 3초 전 | 0.5초 전 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ORDER:
        s=data['statistics'][k]
        vals=s['complete30_medians']
        lines.append('| '+NAMES[k]+f' | {s["complete30_n"]:,} | '+' | '.join(f'{vals[str(t)]:.2f}' for t in (30,20,10,6,3,.5))+' |')
    lines += ['', '앞선 단절이 섞인 효과를 구분하기 위해 앞선 30초에 지속 단절이 없었던 사건도 별도로 확인했다.',
        '점진형은 그 부분집합 54건에서 30초 전 중앙값 29.45dB, 10초 전 29.84dB, 3초 전 4.73dB였다. '
        '점진형 전체를 “오래전부터 낮은 신호”라고 일반화할 수 없다. 기타형의 동일 부분집합은 96건이고 '
        '각각 9.34dB, 4.48dB, 2.54dB였다. 유형별 중앙값은 서로 다른 사건의 값이며 단일 비행 경로가 아니다.','',
        '## 0.1초 관측과 재배치의 시간 예산','',
        '- 현재 원본 전파 계산은 0.5초 간격이다. 두 점 사이의 하락을 관찰했으며 정확히 언제, 얼마나 빠르게 줄었는지는 알 수 없다.',
        '- 기존 두 점을 직선으로 5등분하면 없는 전조를 만들 수 있다. 0.1초 진단은 같은 이동 경로에서 전파를 새로 계산해야 한다.',
        '- 실제 예측 입력으로 쓰려면 비컨·ACK의 실제 수신 및 시간표시도 그 해상도를 지원해야 한다. 현재 비컨은 1초 주기다.',
        '- 실제 채택된 선제 재배치 판단과 명령 적용은 3초마다 수행한다. 0.1초 감시만 추가하면 다음 명령 시각까지 거의 3초 기다릴 수 있다.',
        '- 속도 상한 5m/s에서 0.1초에 0.5m, 0.4초에 2m, 0.5초에 2.5m가 이상적 상한이다. 10m 이동에는 최소 2초가 필요하다.',
        '- 실제 기체는 관측·전송·추론·명령 대기와 가감속 지연이 추가된다. 현재 시뮬레이션은 3초 구간을 일정 속도로 이동한다.',
        '- PX4의 공식 회피 문서도 센서 지연과 속도 명령 추종 지연을 별도로 보정하도록 설명한다. [PX4 공식 문서](https://docs.px4.io/v1.16/en/computer_vision/collision_prevention#cp_delay-delay-tuning)',
        '', '필요한 선행 시간 ≥ 관측 지연 + 계산·전송 지연 + 명령 대기 + 안전한 대체 위치까지 이동 시간 + 여유 시간. '
        '급락을 본 뒤의 짧은 이동만으로 예방할 수 있는 사건이 얼마나 되는지는 별도의 실제 궤적 대조가 필요하다.','',
        '## 후속 검증 방향','',
        '1. 점진형에는 10·20·30초의 수준·감소 추세·경계 도달 예상시간을 후보 특징으로 검증한다. '
        '기타형에는 낮은 여유가 지속된 시간·최근 반복 단절·대체 경로 수를 함께 검증한다.',
        '2. 급락형에는 이동 예정 경로가 건물 차폐 경계에 닿는 시간과, 안전한 중계 위치까지 필요한 이동 시간을 우선 계산한다. '
        '0.1초 감시는 그 예측을 더 자주 수정하는 보조로 비교한다.',
        '3. 0.1초 재계산은 전파·그림자 상관·실제 관측 갱신·명령 지연을 정합적으로 구성한다. '
        '현재 건물 교차와 지붕 회절 근사는 경계에서 손실이 갑자기 바뀔 수 있으므로, 해상도를 높여도 완만한 전조가 생긴다고 가정하지 않는다.',
        '4. 정상 구간까지 포함한 별도 개발·확인 자료에서 같은 오경보 비용을 맞추고, 3초 이상 선행 탐지와 실제 단절 시작률을 각각 대조한다.',
        '', '이번 작업은 기록 확장과 원인 진단이다. 0.1초 실험·추가 학습·제어 성능 향상은 아직 측정하지 않았다. '
        '이전에 진단에 공개한 최종 지도도 포함하므로 새로운 미공개 평가 자료로 사용할 수 없다.','']
    for p in images:
        lines+=['!['+p.stem+']('+p.name+')','']
    path=OUT/'30초_이력확장_분석.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    return path


def main():
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    images=figures(data)
    note=report(data, images)
    template=(ROOT/'tmp/outage_history_template.html').read_text(encoding='utf-8')
    # No workstation paths or model secrets are included in this private display bundle.
    events=[e for f in data['flights'] for e in f['events']]
    display=dict(events=events,statistics=data['statistics'],examples={k:dict(region=v['region'],onset_s=v['onset_s'])
        for k,v in data['examples'].items()}, verified_frames=data['verified_radio_frames'], created_at=data['created_at'])
    html=template.replace('__HISTORY_DATA__',json.dumps(display,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    page=OUT/'history.html'
    page.write_text(html,encoding='utf-8')
    site=ROOT/'sites/final-v1-replay/dist'
    for p in [page,*images,note]:
        shutil.copy2(p,site/p.name)
    # Re-downloadable source contains only diagnostic data, not local archive paths.
    (site/'history-data.json').write_text(json.dumps(display,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    provenance=dict(data_sha256=digest(SOURCE),note_sha256=digest(note),page_sha256=digest(page),
                    figures={p.name:digest(p) for p in images},builder_sha256=digest(__file__),
                    display_event_count=len(events))
    (OUT/'delivery.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(provenance,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
