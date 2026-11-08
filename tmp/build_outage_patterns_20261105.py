"""Render verified outage-pattern summaries from the frozen Final V1 audit."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from final_v1.causes import pattern

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUTPUT = WORKSPACE/"output"/"outage_patterns_20261105"
SOURCE = WORKSPACE/"final_v1"/"artifacts"/"causes.json"
LABELS = {"abrupt":"급락형", "mixed":"혼합형", "gradual":"점진형",
          "other":"기타형", "insufficient_history":"이력 부족"}
COLORS = {"abrupt":"#bc5636", "mixed":"#b4842b", "gradual":"#168472",
          "other":"#657b8e", "insufficient_history":"#aab3ba"}
ORDER = ("abrupt","mixed","gradual","other","insufficient_history")
KIND_NAMES = {"clear":"무간섭", "intermittent":"간헐 간섭", "continuous":"지속 간섭"}
font = FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
plt.rcParams.update({"font.family":font.get_name(), "axes.unicode_minus":False,
                     "font.size":11, "figure.facecolor":"white", "axes.facecolor":"white",
                     "axes.titleweight":"bold", "axes.spines.top":False, "axes.spines.right":False,
                     "savefig.facecolor":"white"})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def distribution(data, groups):
    names = ("abrupt","mixed","other","gradual","insufficient_history")
    count = [len(groups[k]) for k in names]
    total = data["new_outages"]
    fig, ax = plt.subplots(figsize=(12.4,5.8))
    fig.subplots_adjust(left=.14,right=.95,top=.76,bottom=.17)
    fig.text(.07,.94,"통신 단절 직전, 어떤 변화가 나타났나",fontsize=21,fontweight="bold")
    fig.text(.07,.875,f"345개 지도 · 새 지속 단절 {total:,}건 · 0.5초 간격 · 단절 전 6초 분석",
             fontsize=12,color="#435e70")
    for i,k in enumerate(names):
        ax.barh(i,count[i],color=COLORS[k],height=.64)
        ax.text(count[i]+27,i,f"{count[i]:,}건  ·  {count[i]/total:.2%}",va="center",
                fontsize=13,fontweight="bold",color="#203b4c")
    ax.set_yticks(np.arange(len(names)),[LABELS[k] for k in names])
    ax.invert_yaxis()
    ax.set_xlim(0,2150)
    ax.set_xlabel("새 단절 사건 수(건)",labelpad=10)
    ax.grid(axis="x",alpha=.13)
    ax.set_axisbelow(True)
    acute = len(groups["abrupt"])+len(groups["mixed"])
    fig.text(.07,.055,f"급락을 포함한 두 유형 합계  {acute:,}/{total:,}건  ·  {acute/total:.2%}",
             fontsize=13,fontweight="bold",color=COLORS["abrupt"])
    fig.text(.98,.055,"시작부터 단절된 14구간은 이 분류에서 제외",ha="right",fontsize=9,color="#526575")
    path = OUTPUT/"01_pattern_distribution.png"
    fig.savefig(path,dpi=170)
    plt.close(fig)
    return path


def curves(groups, examples):
    x = np.arange(-6.,.01,.5)
    names = ("abrupt","gradual","mixed","other")
    fig, axes = plt.subplots(2,2,figsize=(14.2,10.3))
    fig.subplots_adjust(left=.075,right=.975,top=.86,bottom=.20,hspace=.67,wspace=.20)
    fig.text(.055,.96,"유형별 실제 통신 여유 변화",fontsize=21,fontweight="bold")
    fig.text(.055,.919,"0초는 실제 지속 단절의 시작 시각 · 굵은 선은 실제 대표 사건 1건",
             fontsize=12,color="#435e70")
    lowest, highest = [],[]
    for k in names:
        all_rows = np.array([e["margin_db"] for _,e in groups[k]])
        q10,med,q90 = np.quantile(all_rows,[.1,.5,.9],axis=0)
        lowest.extend(q10)
        highest.extend(q90)
        lowest.extend(examples[k]["margin_db"])
        highest.extend(examples[k]["margin_db"])
    low = math.floor((min(lowest)-4)/5)*5
    high = math.ceil((max(highest)+4)/5)*5
    for ax,k in zip(axes.ravel(),names):
        rows = np.array([e["margin_db"] for _,e in groups[k]])
        q10,med,q90 = np.quantile(rows,[.1,.5,.9],axis=0)
        selected = examples[k]
        ax.axvspan(-.5,0.,color="#dd8e7b",alpha=.12)
        ax.fill_between(x,q10,q90,color=COLORS[k],alpha=.15,linewidth=0)
        ax.plot(x,med,"--",color="#425969",lw=1.5)
        ax.plot(x,selected["margin_db"],"o-",color=COLORS[k],lw=2.5,markersize=4)
        ax.axhline(0,color="#777",lw=1.,ls=":")
        ax.axvline(0,color="#a74e3f",lw=1.,ls=":")
        ax.set_xlim(-6.1,.2)
        ax.set_ylim(low,high)
        ax.set_xticks(np.arange(-6,1))
        ax.set_xlabel("단절 시작 기준 시간(초)")
        ax.set_ylabel("수색망 통신 여유(dB)")
        ax.grid(alpha=.12)
        ax.set_title(f"{LABELS[k]}  ·  {len(groups[k]):,}건",loc="left",pad=30,color=COLORS[k])
        caption = f'{selected["region"]} · {KIND_NAMES[selected["kind"]]} · 시작 {selected["onset_s"]:g}초'
        ax.text(0.,1.035,caption,transform=ax.transAxes,fontsize=9.5,color="#425969")
        ax.text(0.,-.29,f'마지막 0.5초 감소 {selected["last_drop_db"]:.2f}dB  |  초반→후반 중앙값 감소 '
                f'{selected["median_decline_db"]:.2f}dB',transform=ax.transAxes,fontsize=9.5)
    handles = [Line2D([0],[0],color="#334b5b",lw=2.5,marker="o",markersize=4,label="대표 실제 사건"),
               Line2D([0],[0],color="#425969",lw=1.5,ls="--",label="유형의 시각별 중앙값"),
               Patch(facecolor="#9fb2bf",alpha=.35,label="유형 내 10~90백분위 범위")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
    fig.text(.5,.022,"시뮬레이터 전파값으로 복원한 전체망의 여유 · 0dB는 대칭 SINR 기준의 근사 통신 경계",
             ha="center",fontsize=10,color="#526575")
    path = OUTPUT/"02_actual_pattern_curves.png"
    fig.savefig(path,dpi=170)
    plt.close(fig)
    return path


def cause_and_sensitivity(data, groups):
    fig, (a,b) = plt.subplots(1,2,figsize=(14,6.2),gridspec_kw={"width_ratios":[1,1.25]})
    fig.subplots_adjust(left=.08,right=.97,top=.77,bottom=.25,wspace=.38)
    fig.text(.045,.95,"시간 변화의 형태와 전파 원인을 함께 보기",fontsize=20,fontweight="bold")
    names = ("abrupt","mixed","gradual","other")
    for i,k in enumerate(names):
        n = len(groups[k])
        repaired = sum(["building"] in e["repair_factors"] for _,e in groups[k])
        pct = repaired/n*100.
        a.barh(i,pct,color=COLORS[k],height=.57)
        a.text(min(pct,99)+1.8,i,f"{repaired:,}/{n:,} · {pct:.1f}%",va="center",fontsize=10.5)
    a.set_yticks(np.arange(4),[LABELS[k] for k in names])
    a.invert_yaxis()
    a.set_xlim(0,135)
    a.set_xticks([0,25,50,75,100])
    a.set_title("건물의 추가 전파 손실만 제거하면\n단절 첫 1초가 복구된 사건",fontsize=12,loc="left",pad=18)
    a.set_xlabel("각 유형 내 비율(%)")
    a.grid(axis="x",alpha=.12)
    a.set_axisbelow(True)
    conditions = (("sensitivity_4_2","4dB / 2dB"),("pattern","6dB / 3dB"),
                  ("sensitivity_8_4","8dB / 4dB"))
    for row,(source,label) in enumerate(conditions):
        offset=0.
        values=data["sensitivity"][source]
        for k in ORDER:
            pct = values[k]/data["new_outages"]*100.
            b.barh(row,pct,left=offset,color=COLORS[k],height=.54)
            if pct>=7:
                b.text(offset+pct/2,row,f"{pct:.1f}%",ha="center",va="center",fontsize=10.5,color="white",
                       fontweight="bold")
            offset+=pct
        acute=(values["abrupt"]+values["mixed"])/data["new_outages"]*100.
        b.text(102,row,f"급락 포함\n{acute:.1f}%",va="center",fontsize=10,color="#a04c32")
    b.set_yticks(np.arange(3),[label for _,label in conditions])
    b.invert_yaxis()
    b.set_xlim(0,120)
    b.set_xticks([0,25,50,75,100])
    b.set_title("분류 문턱을 바꾼 대조\n행 이름은 급락 기준 / 중앙값 감소 기준",fontsize=12,loc="left",pad=18)
    b.set_xlabel("전체 새 단절 중 비율(%)")
    handles=[Patch(facecolor=COLORS[k],label=LABELS[k]) for k in ORDER]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.115),ncol=5,frameon=False)
    fig.text(.5,.045,"건물 항 제거는 원인 조사용 가상 대조다. 같은 사건이 간섭·섀도잉 제거로도 복구될 수 있다.",
             ha="center",fontsize=10,color="#526575")
    path=OUTPUT/"03_causes_and_thresholds.png"
    fig.savefig(path,dpi=170)
    plt.close(fig)
    return path


def report(data,groups,examples,stats,figures):
    total=data["new_outages"]
    lines=["# Final V1 · 급락형·점진형·혼합형 단절 분석","",
           "345개 지도에서 V7 PPO가 수행한 345개 비행을 대상으로, 새로 시작한 지속 단절 2,572건을 분류했다. "
           "각 지도에는 비행 방향과 난수 한 세트를 배정했다. 무간섭·간헐 간섭·지속 간섭 중 실제 제어 비행에 "
           "사용한 조건의 기록을 분석했다. 시작 시점부터 연결이 끊긴 14구간은 새 단절 분모에서 제외했다.","",
           "## 통신 여유와 단절의 의미","",
           "수색기 한 쌍을 잇는 직접 링크와 중계 경로를 모두 고려한다. 경로에 포함된 가장 약한 링크의 "
           "여유가 가장 큰 경로를 각 수색기 쌍에서 고르고, 그중 가장 약한 쌍의 값을 수색망 통신 여유로 "
           "정의했다. 단위는 dB다. 양수일수록 통신 경계까지 여유가 있고 음수는 기준 미달을 뜻한다.","",
           "그래프의 여유는 방향별 SINR과 대칭 임계값으로 근사한 값이다. 실제 지속 단절은 데이터·ACK "
           "성공확률 기준의 링크로 수색망이 1초 이상 연결되지 않는 구간이다. 해당 실제 판정은 원본 기록과 "
           "동일하게 재현했다.","",
           "## 정확한 분류 기준","",
           "각 사건에서 단절 시작을 0초로 맞추고 -6초부터 0초까지 0.5초 간격의 13개 값을 사용했다. "
           "두 양을 계산했다.","",
           "- 마지막 급락량 J는 -0.5초의 여유에서 0초의 여유를 뺀 값이다.",
           "- 앞선 감소량 D는 -6~-4.5초의 네 값 중앙값에서 -2~-0.5초의 네 값 중앙값을 뺀 값이다.","",
           "| 유형 | 정확한 조건 | 의미 | 건수 | 비율 |",
           "|---|---|---|---:|---:|"]
    descriptions={
        "abrupt":("J ≥ 6dB, D < 3dB","앞선 구간의 감소 기준을 충족하지 않고 마지막에 크게 하락"),
        "mixed":("J ≥ 6dB, D ≥ 3dB","앞서 여유가 줄고 마지막 0.5초에도 크게 하락"),
        "gradual":("J < 6dB, D ≥ 3dB","앞서 여유가 줄며 마지막 급락 기준은 충족하지 않음"),
        "other":("J < 6dB, D < 3dB","두 감소 기준은 작지만 실제 지속 단절이 발생"),
        "insufficient_history":("시작 전 6초 자료 부족","같은 길이의 이력으로 분류하기 어려움"),
    }
    for k in ORDER:
        rule,meaning=descriptions[k]
        lines.append(f"| {LABELS[k]} | {rule} | {meaning} | {len(groups[k]):,} | {len(groups[k])/total:.2%} |")
    lines+=["", "6dB와 3dB는 이 분석에서 정한 연구용 분류 문턱이다. 점진형은 두 구간의 중앙값 감소를 "
            "뜻하므로 모든 0.5초 구간에서 일정하게 감소한다는 의미를 포함하지 않는다. 혼합형은 시간 변화의 "
            "두 조건을 함께 만족한 경우다. 물리 원인이 두 가지 이상이라는 뜻은 별도의 원인 조사에서 판단한다.","",
            f"급락형과 혼합형 합계는 2,185/2,572건, 84.95%다. 이 비율은 단절 시작 시점까지 포함한 "
            "사후 분류다. 단절 3초 전에 같은 급락을 관측했다는 비율로 해석할 수 없다.","",
            f"![유형 분포]({figures[0].name})","",
            "## 실제 대표 사례","",
            "각 유형의 시각별 중앙값에 가장 가까운 실제 사건을 선택했다. 각 시각의 사분위 범위로 "
            "차이를 정규화한 평균 절대거리를 사용했다. 유형 전체의 중앙값과 10~90백분위 범위도 함께 표시했다. "
            "원본 파형을 매끈하게 만들지 않고 저장된 반초 값을 연결했다. 사건 시작 전 6초에는 "
            "앞선 별도 단절과 회복이 포함될 수 있어 일부 유형의 10백분위가 0dB 아래에 있다. "
            "J와 D의 두 조건으로 현재 사건의 유형을 정했다.","",
            f"![유형별 실제 변화 곡선]({figures[1].name})","",
            "| 유형 | 대표 지도 | 간섭 조건 | 단절 시작 | J | D | 지속 시간 |",
            "|---|---|---|---:|---:|---:|---:|"]
    for k in ("abrupt","gradual","mixed","other"):
        e=examples[k]
        lines.append(f"| {LABELS[k]} | {e['region']} | {KIND_NAMES[e['kind']]} | {e['onset_s']:g}초 | "
                     f"{e['last_drop_db']:.2f}dB | {e['median_decline_db']:.2f}dB | {e['duration_s']:g}초 |")
    lines+=["", "급락형 대표는 약 30dB의 여유를 유지하다 마지막에 -12.40dB로 내려간다. 점진형 대표는 "
            "9.28dB에서 0.28dB까지 낮아진 뒤 -1.01dB에 도달한다. 혼합형 대표는 초반 약 33dB에서 "
            "후반 약 28dB로 약해지고 마지막에 -14.69dB가 된다. 기타형 대표는 처음부터 약 1dB의 작은 "
            "여유를 갖고 있다가 0dB 아래로 내려간다. 큰 하락이 없어도 통신 경계 가까이에서 단절할 수 있다.","",
            "## 유형별 지속 시간과 지도 수","",
            "| 유형 | 사건이 나온 지도 수 | 지속 시간 중앙값 | 지속 시간 90백분위 |",
            "|---|---:|---:|---:|"]
    for k in ORDER:
        s=stats[k]
        lines.append(f"| {LABELS[k]} | {s['maps']} | {s['duration_median_s']:g}초 | {s['duration_p90_s']:.2f}초 |")
    lines+=["", "같은 지도가 여러 유형에 포함될 수 있다. 90백분위는 해당 유형 사건의 90%가 그 시간 "
            "이내에 끝났음을 뜻한다. 파형이 어떻게 떨어졌는지와 단절이 얼마나 오래 지속됐는지는 구분한다.","",
            "## 물리 원인과의 관계","",
            "| 유형 | 건물 추가손실만 제거해 첫 1초 복구 | 간섭만 제거해 복구 | 섀도잉만 제거해 복구 |",
            "|---|---:|---:|---:|"]
    for k in ORDER:
        n=len(groups[k])
        factors=stats[k]["single_factor_repair"]
        lines.append(f"| {LABELS[k]} | {factors.get('building',0)}/{n} ({factors.get('building',0)/n:.1%}) | "
                     f"{factors.get('interference',0)}/{n} | {factors.get('shadow',0)}/{n} |")
    lines+=["", "같은 실제 위치와 외생 상태를 두고 각 손실 항을 제거한 가상 대조다. 한 사건이 건물 항 "
            "제거와 간섭 제거에서 각각 복구될 수 있으므로 행의 세 원인 수는 겹친다. 한 가지 원인에 "
            "사건을 독점 배정한 표가 아니다.","",
            "급락형 1,676건 중 1,670건과 혼합형 509건 전부는 건물 항만 제거해 첫 1초가 복구됐다. "
            "이 시뮬레이션에서는 이동 경로가 건물 차폐에 들어가는 지점을 미리 계산하는 접근을 뒷받침한다. "
            "점진형도 101건 중 71건이 같은 대조에서 복구됐으므로 시간 변화 형태 하나만으로 전파 원인을 "
            "단정할 수 없다.","",
            f"![원인 대조와 문턱 변화]({figures[2].name})","",
            "## 문턱에 따른 민감도","",
            "| 급락 J / 감소 D 문턱 | 급락형 | 혼합형 | 점진형 | 기타형 | 이력 부족 | 급락 포함 비율 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for field,title in (("sensitivity_4_2","4 / 2dB"),("pattern","6 / 3dB"),("sensitivity_8_4","8 / 4dB")):
        c=data["sensitivity"][field]
        lines.append(f"| {title} | {c['abrupt']:,} | {c['mixed']:,} | {c['gradual']:,} | {c['other']:,} | "
                     f"{c['insufficient_history']} | {(c['abrupt']+c['mixed'])/total:.2%} |")
    lines+=["", "문턱을 바꾸면 급락형과 혼합형 사이에서 사건이 이동한다. 급락을 포함한 합계는 세 기준에서 "
            "약 84~86%였다. 개별 유형의 정확한 비율은 문턱을 함께 명시해야 한다.","",
            "## 선행 예측으로 연결할 때의 해석","",
            "이 파형은 시뮬레이터에서 복원한 전체 수색망의 실제 전파 변화다. 예측기가 받을 수 있는 "
            "자료는 그 시점까지 실제로 수신한 비컨·패킷·ACK다. 전체망의 급락이 다른 위치에서 일어나거나 "
            "급락과 함께 보고가 사라지면 첫 중계기가 같은 모양을 관측하지 못할 수 있다.","",
            "별도의 관측 변화 검증은 알려진 지도 14개의 새 난수 42비행·216사건에서 수행했다. 첫 중계기의 "
            "단절 3초 전 관측에서 직전 0.5초 6dB 이상 감소는 8/216건이었다. 현재 3초 제어 경계에서는 "
            "1초 주기 비컨의 0.5초 차분이 모두 결측이었고 ACK의 차분은 2.3% 시점에서 가능했다.","",
            "기존 예정 경로·차폐 특징에 실제 관측 변화량을 추가한 모델과 같은 설정의 기존 특징 모델은 "
            "문턱 세분화 대조에서 모두 119/216건, 55.09%를 선행 탐지했다. 따라서 이번 데이터에서는 "
            "0.5초 급락 빈도를 곧바로 높은 선행 탐지율로 연결할 근거가 확보되지 않았다. 유형별 파형은 "
            "예정 경로에서 차폐될 위치를 미리 계산할 필요성과, 미리 확보한 관측 정보의 범위를 함께 "
            "점검하는 데 사용할 수 있다.","",
            "## 근거와 재현 범위","",
            f"- 원자료는 Final V1의 causes.json이며 SHA-256은 {digest(SOURCE)}다.",
            "- 분류 코드의 해시, 유형별 합계, 문턱별 합계와 실제 사례의 분류를 확인했다.",
            "- 저장된 파형은 소수 둘째 자리로 반올림돼 있다. 이 값으로 재분류했을 때도 원래 분류와 모두 일치했다.",
            "- 그래프의 대표 사건은 유형 전체의 통계와 구분하며 전체 수신 환경의 대표성을 보장하는 표본으로 해석하지 않는다.",
            "- 지도별 비행 방향과 난수 한 세트의 시뮬레이션 분석이다. 실제 장비 시험과 별도다.",
            "", "[도표용 집계와 대표 사건](evidence.json)",""]
    out=OUTPUT/"급락형_점진형_혼합형_상세분석.md"
    out.write_text("\n".join(lines),encoding="utf-8")
    return out


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=json.loads(SOURCE.read_text(encoding="utf-8"))
    release=json.loads((WORKSPACE/"final_v1"/"artifacts"/"release.json").read_text(encoding="utf-8"))
    assert digest(SOURCE)==release["evidence_sha256"][str(SOURCE)]
    groups=defaultdict(list)
    left=0
    for flight in data["flights"]:
        for event in flight["events"]:
            if event["onset_s"]==0:
                left+=1
            else:
                groups[event["pattern"]].append((flight,event))
    assert left==data["initially_down"]==14
    assert sum(map(len,groups.values()))==data["new_outages"]==2572
    assert data["maps"]==len(data["flights"])==345
    examples,stats={},{}
    rounded_disagreements=0
    for k in ORDER:
        rows=groups[k]
        assert len(rows)==data["by_pattern"][k]["events"]
        factors=Counter(part[0] for _,event in rows for part in event["repair_factors"] if len(part)==1)
        duration=np.array([e["end_s"]-e["onset_s"] for _,e in rows])
        stats[k]=dict(events=len(rows),fraction=len(rows)/2572,
                      maps=len({f["region"] for f,_ in rows}),
                      duration_median_s=float(np.median(duration)),
                      duration_p90_s=float(np.quantile(duration,.9)),
                      single_factor_repair=dict(factors))
        if k=="insufficient_history":
            continue
        a=np.array([e["margin_db"] for _,e in rows])
        median=np.median(a,axis=0)
        scale=np.maximum(1.,np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0))
        distance=np.mean(np.abs(a-median)/scale,axis=1)
        for trace in a:
            rounded_disagreements+=pattern(trace)!=k
        selected=int(np.argsort(distance,kind="stable")[0])
        f,e=rows[selected]
        assert pattern(a[selected])==k
        examples[k]=dict(region=f["region"],kind=f["kind"],onset_s=e["onset_s"],
                         duration_s=e["end_s"]-e["onset_s"],margin_db=e["margin_db"],
                         last_drop_db=float(a[selected,-2]-a[selected,-1]),
                         median_decline_db=float(np.median(a[selected,:4])-np.median(a[selected,-5:-1])),
                         original_repairs=e["repair_factors"])
    assert rounded_disagreements==0
    figures=[distribution(data,groups),curves(groups,examples),cause_and_sensitivity(data,groups)]
    out=report(data,groups,examples,stats,figures)
    evidence=dict(source_sha256=digest(SOURCE),rules=data["rule"],maps=345,new_events=2572,
                  initially_down=14,statistics=stats,examples=examples,sensitivity=data["sensitivity"],
                  rounded_classification_disagreements=rounded_disagreements,
                  figure_sha256={p.name:digest(p) for p in figures},report_sha256=digest(out))
    write_json(OUTPUT/"evidence.json",evidence)
    print(json.dumps(dict(report=str(out),figures=[str(p) for p in figures],counts={k:len(groups[k]) for k in ORDER}),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
