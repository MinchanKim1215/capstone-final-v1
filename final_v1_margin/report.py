"""Timing-specific pre-signal evidence and independent predictor comparison."""
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from final_v1.campaign import native_patterns
from forecast_learning.features import COLUMNS
from forecast_rebuild.contracts import verify_prior

from . import ROOT,ARTIFACTS
from .features import CHANNELS,HALF_SECOND_COLUMNS,NAMES,TWO_SECOND_COLUMNS,maximum_drop
from .protocol import V21,maps,now,read,sha,status,write
from .replay import LEADS
from .training import FAMILIES,case,load,metrics,predictions,restore,window

LABELS = dict(physics_early="기존 물리 특징 재학습",rf_early="관측 변화만 학습",
              current_early="기존 특징 + 현재 관측값",
              augmented_early="기존 특징 + 관측 변화",rf_imminent="관측 변화 · 3초 미만",
              legacy="기존 V21 점수",drop_rule="관측 감소량 경보",half_second_rule="관측 0.5초 · 6dB 감소")


def paired_recall(flights,a,b,threshold_a,threshold_b,win=(3.,7.)):
    registry = maps()
    groups = defaultdict(list)
    for f,pa,pb in zip(flights,a,b):
        ca,cb = case(f,pa,threshold_a,win),case(f,pb,threshold_b,win)
        group = registry[f.row["spec"]["region"]]["joint_spatial_group"]
        groups[group].append([ca["events"],ca["hits"],cb["hits"]])
    values = np.array([np.sum(parts,axis=0) for parts in groups.values()])
    rng = np.random.default_rng(2026110502)
    selected = values[rng.integers(len(values),size=(4000,len(values)))].sum(axis=1)
    delta = (selected[:,2]-selected[:,1])/np.maximum(1,selected[:,0])
    total = values.sum(axis=0)
    return dict(recall_delta_pp=float((total[2]-total[1])/max(1,total[0])*100.),
                ci95_pp=(np.quantile(delta,[.025,.975])*100.).tolist(),groups=len(groups))


def signal_summary(flights,shapes):
    groups = defaultdict(list)
    aligned = np.concatenate([f.aligned for f in flights])
    for f,kinds in zip(flights,shapes):
        if len(kinds) != len(f.onsets):
            raise ValueError("Cut classification count differs from RF event ledger")
        for kind,event in zip(kinds,f.aligned):
            groups[kind].append(event)

    def stats(values):
        if not len(values):
            return []
        result = []
        for j,lead in enumerate(LEADS):
            feature = values[:,j]
            all_drop = maximum_drop(feature)
            half = maximum_drop(feature,HALF_SECOND_COLUMNS)
            two = maximum_drop(feature,TWO_SECOND_COLUMNS)
            result.append(dict(lead_s=lead,events=len(values),
                               any_distinct_measurement=int(np.isfinite(all_drop).sum()),
                               half_second_distinct_measurement=int(np.isfinite(half).sum()),
                               half_second_drop_6db=int((half>=6.).sum()),
                               two_second_drop_3db=int((two>=3.).sum()),
                               any_drop_3db=int((all_drop>=3.).sum()),
                               finite_median_max_drop_db=float(np.nanmedian(all_drop)) if np.isfinite(all_drop).any() else None))
        return result
    coverage = {}
    for channel in CHANNELS:
        coverage[channel] = {}
        for lag in (.5,1.,2.):
            column = NAMES.index(f"{channel}_{lag:g}s_max_drop")
            count = sum(np.isfinite(f.rf[:,column]).sum() for f in flights)
            total = sum(len(f.rf) for f in flights)
            coverage[channel][str(lag)] = dict(distinct_measurement_queries=int(count),queries=total,
                                               fraction=float(count/total))
    return dict(all_events=stats(aligned),by_pattern={k:stats(np.asarray(v)) for k,v in groups.items()},
                decision_query_coverage=coverage)


def plot(results,signal,refined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10})
    fig,axes = plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    fig.suptitle("Final V1 · 실제 관측 변화의 선행 예측 검증",fontsize=16,fontweight="bold")
    rows = signal["all_events"]
    x = np.array([r["lead_s"] for r in rows])
    n = np.array([r["events"] for r in rows])
    axes[0].plot(x,np.array([r["two_second_drop_3db"] for r in rows])/n*100.,"o-",color="#1b8878",
                 label="직전 2초에 3dB 이상 감소")
    axes[0].plot(x,np.array([r["half_second_drop_6db"] for r in rows])/n*100.,"s-",color="#be6d22",
                 label="직전 0.5초에 6dB 이상 감소")
    axes[0].axvspan(3,7,color="#cbdde8",alpha=.4)
    axes[0].set(xlabel="실제 단절까지 남은 시간(초)",ylabel="전체 사건 중 관측된 비율(%)",title="실제로 받은 정보에 전조가 있었나")
    axes[0].set_xticks(x)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False,fontsize=9)
    variants = ("legacy","physics_early","current_early","rf_early","augmented_early")
    vals = [refined[k] for k in variants]
    positions = np.arange(len(vals))
    axes[1].barh(positions,[100*r["recall"] for r in vals],color=["#526b7c","#698da1","#c58238","#1b8878"])
    for i,r in enumerate(vals):
        axes[1].text(100*r["recall"]+.7,i,f'{r["recall"]:.1%} · 오경보 {r["false_per_hour"]:.2f}/h',
                     va="center",fontsize=9)
    axes[1].set_yticks(positions,[LABELS[k] for k in variants])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="3초 이상 7초 미만 선행 탐지율(%)",title="같은 비행 · 문턱 세분화 대조")
    axes[1].set_xlim(0,max(65,max(100*r["recall"] for r in vals)+36))
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    file = ROOT/"관측변화_검증.png"
    fig.savefig(file,dpi=160,facecolor="white")
    plt.close(fig)
    return file


def describe(plan,freeze,result):
    lines = ["# Final V1 · 관측 통신 여유 변화 검증", "",
             f"검증 완료 {result['completed_at']}", "",
             "실제 수신한 통신 여유의 최근 0.5초·1초·2초 감소량과 기울기 변화를 추가했을 때 "
             "3초 이상 선행 예측이 개선되는지 확인했다.", "",
             "## 관측과 검증 방법", "",
             "첫 중계기가 받은 비컨의 수신 SINR, 상대가 보고한 역방향 SINR, 성공한 데이터 전송의 ACK SINR을 "
             "사용했다. 마지막 수신 시각이 다른 두 측정만 차분했다. 같은 측정의 복사값, 마스크가 꺼진 값, "
             "2초보다 오래된 값은 변화량으로 인정하지 않았다. 비컨은 1초 주기이고 ACK 관측은 성공한 "
             "전송이 있을 때 0.5초 구간에서도 갱신된다.", "",
             f"훈련 245개 지도, {freeze['training_rows']:,}개 시점. 기존 특징 214개와 새 관측 특징 "
             f"{freeze['new_feature_count']}개를 사용했다. 주요 모델의 학습 자료와 알고리즘 설정을 맞췄다. "
             "현재 신호 수준·수신 여부·나이 12개만 추가한 대조 모델을 확인 비행 전에 별도로 등록했다. "
             "보정 45비행에서 문턱을 정하고 별도 선정 45비행에서 갱신 횟수를 선택했다. 새 난수 42조건을 "
             "열기 전에 모두 고정했다. 확인은 알려진 지도 14개에서 수행했다.", "",
             "## 단절 전에 실제 관측된 변화", "",
             "| 단절까지 남은 시간 | 사건 수 | 새 측정끼리 비교 가능 | 직전 2초에 3dB 이상 감소 | 직전 0.5초에 6dB 이상 감소 |",
             "|---|---:|---:|---:|---:|"]
    for row in result["signal"]["all_events"]:
        n = row["events"]
        lines.append(f"| {row['lead_s']:g}초 | {n} | {row['any_distinct_measurement']}/{n} | "
                     f"{row['two_second_drop_3db']}/{n} ({row['two_second_drop_3db']/n:.1%}) | "
                     f"{row['half_second_drop_6db']}/{n} ({row['half_second_drop_6db']/n:.1%}) |")
    lines += ["", "이 표는 단절 이전의 서로 다른 관측 시각에서 계산했다. 실제 단절 순간의 전파 정답은 "
              "특징에 넣지 않았다. 3dB·6dB 조건은 관측 패턴을 요약하는 고정 기준이며 예측 성공률과 구분한다.", "",
              "| 실제 3초 제어 경계의 관측 | 0.5초 차분 가능 | 1초 차분 가능 | 2초 차분 가능 |",
              "|---|---:|---:|---:|"]
    channel_labels = dict(beacon_inbound="수신 비컨",beacon_bidirectional="양방향 비컨 정보",ack_inbound="패킷 ACK")
    for channel,coverage in result["signal"]["decision_query_coverage"].items():
        fractions = [f"{coverage[str(lag)]['fraction']:.1%}" for lag in (.5,1.,2.)]
        lines.append(f"| {channel_labels[channel]} | {' | '.join(fractions)} |")
    lines += ["", "새 측정이 없는 경우의 차분은 결측이다. 비컨 주기와 제어 경계가 동기화돼 있으면 "
              "마지막 0.5초 동안 새 비컨이 없을 수 있다. 단절 시각에 정렬한 관측 표와 실제 제어 시각의 "
              "입력 가능성을 함께 봐야 한다.", "",
              "## 같은 궤적에서 예측 성능", "",
              "| 방법 | 제때 탐지 | 선행 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | AP |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,r in result["metrics"].items():
        ap = f"{r['ap']:.2%}" if r["ap"] is not None else "자료 없음"
        precision = f"{r['precision']:.2%}" if r["alerts"] else "경보 없음"
        lines.append(f"| {LABELS[name]} | {r['hits']}/{r['events']} | {r['recall']:.2%} | {precision} | "
                     f"{r['false_per_hour']:.2f} | {ap} |")
    lines += ["", "제때 탐지는 3초 이상 7초 미만 전에 시작한 경보를 실제 사건과 한 번씩 매칭한 값이다. "
              "관측 변화 · 3초 미만 행은 0.5초 이상 3초 미만의 임박 탐지로 평가했다. 해당 행의 수치를 "
              "3초 선행 탐지율로 사용할 수 없다. 오경보 한도는 보정 자료에서 4회/h로 정했고, "
              "확인 비행에서 실제로 지켜졌는지 함께 보고했다.", "",
              "| 비교 | 선행 탐지율 차이 | 공간 그룹 95% 구간 |", "|---|---:|---:|"]
    for name,label in (("vs_physics","관측 추가 vs 같은 설정의 기존 특징"),
                       ("vs_current","변화량 포함 vs 현재 관측값만 추가"),("vs_legacy","관측 추가 vs V21")):
        d = result[name]
        lines.append(f"| {label} | {d['recall_delta_pp']:+.2f}%p | [{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]%p |")
    lines += ["", f"사전에 정한 추가 입력 채택 기준 충족 여부 **{'충족' if result['accepted'] else '미충족'}**.",
              "기준은 기존 특징 재학습 대비 선행 탐지율 차이의 95% 하한이 양수, 정밀도 감소 2%p 이내, "
              "확인 비행의 오경보·늦은 경보 4회/h 이내다.",
              f"현재 관측값만 추가한 모델과 비교한 변화량의 추가 효과는 "
              f"**{'확인' if result['derivative_specific_support'] else '미확인'}**이다.", "",
              "## 높은 점수 구간의 문턱 세분화 대조", "",
              "초기 문턱 탐색은 99.4백분위 위 구간을 성기게 나눴다. 이 때문에 낮은 오경보 조건에서 가능한 "
              "경보를 놓칠 수 있어 98~99.99백분위 사이를 더 촘촘히 보정한 대조를 추가했다. 모델과 "
              "학습 횟수는 초기 설정을 사용했다. 보정 자료만으로 문턱을 정했으며 새 확인 비행의 성능을 "
              "집계하기 전에 별도 파일로 고정했다.", "",
              "| 방법 | 제때 탐지 | 선행 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 |",
              "|---|---:|---:|---:|---:|"]
    for name,r in result["refined_metrics"].items():
        lines.append(f"| {LABELS[name]} | {r['hits']}/{r['events']} | {r['recall']:.2%} | "
                     f"{r['precision']:.2%} | {r['false_per_hour']:.2f} |")
    lines += ["", "| 문턱 세분화 비교 | 선행 탐지율 차이 | 공간 그룹 95% 구간 |", "|---|---:|---:|"]
    for name,label in (("vs_physics","관측 추가 vs 기존 특징"),("vs_current","변화량 포함 vs 현재 관측값만 추가"),
                       ("vs_legacy","관측 추가 vs V21")):
        d = result["refined_comparisons"][name]
        lines.append(f"| {label} | {d['recall_delta_pp']:+.2f}%p | [{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]%p |")
    lines += ["", f"문턱 세분화 대조의 추가 입력 채택 기준은 "
              f"**{'충족' if result['refined_accepted'] else '미충족'}**이다.",
              f"현재 관측값에 더한 변화량 자체의 추가 효과는 "
              f"**{'확인' if result['refined_derivative_specific_support'] else '미확인'}**이다.", "",
              "## 해석 범위", "",
              "분석 대상은 첫 중계기의 실제 수신 정보다. 다른 위치에서 끊길 링크의 전조가 이 중계기에 도착하지 "
              "않을 수 있다. 원인 분석에 쓰는 전체 수색망의 참 통신 여유와 구분한다. 확률 순위 AP와 "
              "고정 경보 문턱의 선행 탐지율도 별도로 해석한다. 재배치 명령을 바꿔 본 시험은 이 검증 범위에 "
              "포함되지 않는다.", "", "![관측과 예측 비교](관측변화_검증.png)", "",
              "[입력·재현 안내](README.md) · [검증 집계](artifacts/results.json)", ""]
    return "\n".join(lines)


def finish(plan,freeze,records):
    flights = [load(r) for r in records]
    if len(flights)!=42:
        raise ValueError("Independent confirmation incomplete")
    scores,results = {},{}
    for family in (*FAMILIES,"legacy","drop_rule","half_second_rule"):
        choice = freeze["choices"][family]
        model = restore(freeze["models"][family]) if family in FAMILIES else None
        p = predictions(flights,family,model,choice.get("rounds"))
        scores[family] = p
        results[family] = metrics(flights,p,choice["threshold"],window(family),True)
    a = freeze["choices"]["augmented_early"]["threshold"]
    comparisons = {}
    for ref,key in (("physics_early","vs_physics"),("current_early","vs_current"),("legacy","vs_legacy")):
        comparisons[key] = paired_recall(flights,scores[ref],scores["augmented_early"],
                                          freeze["choices"][ref]["threshold"],a)
    refined_plan = read(ARTIFACTS/"threshold_refinement.json")
    if (refined_plan["freeze_sha256"]!=sha(ARTIFACTS/"freeze.json") or
            refined_plan["source_sha256"]!=sha(ROOT/"refine.py") or
            refined_plan["development_manifest_sha256"]!=sha(ARTIFACTS/"development_manifest.json")):
        raise ValueError("Refined thresholds or their development inputs changed")
    refined = {name:metrics(flights,scores[name],info["threshold"],window(name),True)
               for name,info in refined_plan["choices"].items()}
    refined_comparisons = {}
    for ref,k in (("physics_early","vs_physics"),("current_early","vs_current"),("legacy","vs_legacy")):
        refined_comparisons[k] = paired_recall(flights,scores[ref],scores["augmented_early"],
                                                refined_plan["choices"][ref]["threshold"],
                                                refined_plan["choices"]["augmented_early"]["threshold"])
    registry = maps()
    shapes = []
    for f in flights:
        cuts = native_patterns(f.row,registry)
        lookup = {start:shape for start,_,shape in cuts if start>0.}
        shapes.append([lookup[float(start)] for start in f.onsets])
    signal = signal_summary(flights,shapes)
    by_pattern = {}
    for name in ("legacy","physics_early","rf_early","augmented_early"):
        hits = defaultdict(Counter)
        for f,p,kinds in zip(flights,scores[name],shapes):
            c = case(f,p,freeze["choices"][name]["threshold"],(3.,7.))
            for kind,hit in zip(kinds,c["hit_mask"]):
                hits[kind]["events"] += 1
                hits[kind]["hits"] += int(hit)
        by_pattern[name] = hits
    extra = restore(freeze["models"]["augmented_early"])
    importance = extra.feature_importance(importance_type="gain",iteration=freeze["choices"]["augmented_early"]["rounds"])
    names = list(COLUMNS)+list(NAMES)
    order = np.argsort(-importance)[:20]
    gain = float(importance[len(names)-len(NAMES):].sum()/max(1e-12,importance.sum()))
    augmented,base = results["augmented_early"],results["physics_early"]
    accepted = (comparisons["vs_physics"]["ci95_pp"][0]>0. and augmented["precision"]>=base["precision"]-.02
                and augmented["false_per_hour"]<=4.)
    derivative = (comparisons["vs_current"]["ci95_pp"][0]>0. and
                  augmented["precision"]>=results["current_early"]["precision"]-.02 and
                  augmented["false_per_hour"]<=4.)
    new,old,current = (refined[k] for k in ("augmented_early","physics_early","current_early"))
    refined_accept = (refined_comparisons["vs_physics"]["ci95_pp"][0]>0. and
                      new["precision"]>=old["precision"]-.02 and new["false_per_hour"]<=4.)
    refined_derivative = (refined_comparisons["vs_current"]["ci95_pp"][0]>0. and
                          new["precision"]>=current["precision"]-.02 and new["false_per_hour"]<=4.)
    result = dict(version=plan["version"],completed_at=now(),metrics=results,**comparisons,accepted=bool(accepted),
                  signal=signal,by_pattern=by_pattern,new_feature_gain_fraction=gain,
                  derivative_specific_support=bool(derivative),
                  refined_metrics=refined,refined_comparisons=refined_comparisons,
                  refined_accepted=bool(refined_accept),refined_derivative_specific_support=bool(refined_derivative),
                  refined_thresholds_sha256=sha(ARTIFACTS/"threshold_refinement.json"),
                  top_features=[dict(name=names[i],gain=float(importance[i])) for i in order],
                  fresh_flights=len(flights),events=sum(len(f.onsets) for f in flights),
                  thresholds={name:r["threshold"] for name,r in freeze["choices"].items()})
    for path,digest in plan["dependencies"].items():
        if sha(path)!=digest:
            raise ValueError("Pinned Final V1 dependency changed")
    verify_prior()
    image = plot(results,signal,refined)
    write(ARTIFACTS/"results.json",result)
    report = ROOT/"RESULTS.md"
    report.write_text(describe(plan,freeze,result),encoding="utf-8")
    counts = {scope:0 for scope in ("train","calibration","selection","fresh")}
    for stage in ("development","fresh"):
        for record in read(ARTIFACTS/f"{stage}_manifest.json")["records"]:
            f = load(record)
            scope = f.row["contract"]["task"]["scope"]
            counts[scope] += 1
            if not (f.row["future_prefix_invariant"] and f.row["masked_value_invariant"] and f.row["audited"]["passed"]):
                raise ValueError("Causality or native audit failed")
            if scope!="fresh" and not f.row["original_trajectory_and_packets_exact"]:
                raise ValueError("Original trajectory equivalence failed")
    if counts != dict(train=245,calibration=45,selection=45,fresh=42):
        raise ValueError("Final evidence population mismatch")
    write(ARTIFACTS/"completed.json",dict(report_sha256=sha(report),plot_sha256=sha(image),
                                         result_sha256=sha(ARTIFACTS/"results.json"),counts=counts,
                                         threshold_refinement_sha256=sha(ARTIFACTS/"threshold_refinement.json"),
                                         freeze_sha256=sha(ARTIFACTS/"freeze.json"),completed_at=now()))
    status("prediction_and_pre_signal_validation_completed",state="completed",**counts)
    print("completed",accepted,comparisons,flush=True)
