from pathlib import Path
import json
import base64
import hashlib
import html
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final_v1_fast.report import LABELS

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(p):return 'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode('ascii')

def main():
    study=ROOT/'final_v1_fast';art=study/'artifacts'
    result=json.loads((art/'results.json').read_text(encoding='utf-8'))
    done=json.loads((art/'completed.json').read_text(encoding='utf-8'))
    examples=json.loads((art/'paired_examples.json').read_text(encoding='utf-8'))
    out=ROOT/'output/final_v1_fast_summary_20261105';out.mkdir(parents=True,exist_ok=True)
    summary=result['summary'];base=summary['p3_b1'];candidate=summary['p05_b1']
    rows=[]
    for arm,s in summary.items():
        klass='candidate' if arm=='p05_b1' else ''
        rows.append(f'<tr class="{klass}"><td>{html.escape(LABELS[arm])}</td><td>{s["onsets"]}</td><td>{s["cut_seconds"]:.1f}</td><td>{s["timely_pdr"]:.2%}</td><td>{s["airtime_fraction"]:.1%}</td><td>{s["reversals"]}</td></tr>')
    ci=[]
    for arm,c in result['comparisons'].items():
        lo,hi=c['onset_delta_ci95']
        ci.append(f'<tr><td>{html.escape(LABELS[arm])}</td><td>{c["onset_delta"]:+d}</td><td>[{lo:+.0f}, {hi:+.0f}]</td><td>{c["timely_pdr_pp"]:+.2f}%p</td></tr>')
    stress=[]
    for arm,s in result['stress'].items():
        stress.append(f'<tr><td>{html.escape(LABELS[arm])}</td><td>{s["cases"]}</td><td>{s["onsets"]}</td><td>{s["cut_seconds"]:.1f}</td><td>{s["timely_pdr"]:.2%}</td></tr>')
    sample=examples['examples'][0] if examples['examples'] else None
    sample_html=''
    if sample:
        sample_html=f'''<section class="card"><h2>판단 → 계산 → 수신 → 실제 이동</h2>
        <p>설명용으로 사후 선정한 한 사례입니다. 위험 판단 <b>{sample['calculation_start_s']:.1f}초</b>, 계산 완료 <b>{sample['result_ready_s']:.1f}초</b>,
        새 목표 수신 <b>{sample['actual_offset_received_s']:.1f}초</b>, 두 비행의 첫 위치 차이 <b>{sample['first_position_difference_s']:.1f}초</b>였습니다.</p>
        <p>3초 판단의 단절은 {sample['baseline_cut_start_s']:.1f}초에 시작했습니다. 0.5초 판단은 요청 후 12초 동안 연결을 유지했습니다.
        명령을 빨리 받았더라도 기존 안전 간선의 이동을 마치는 시간이 필요했습니다.</p>
        <img src="{image(study/'빠른제어_예방사례.png')}" alt="실제 시뮬레이션의 통신 여유와 판단·수신·이동 시점 대조">
        <p class="note">이 사례는 추가 검증 표본이나 전체 예방률의 근거가 아닙니다. 앞선 위치·링크가 일치하는지 확인한 설명용 사례입니다.</p></section>'''
    payload=dict(created_at=result['created_at'],scope=result['scope'],summary=summary,comparisons=result['comparisons'],
                 stress=result['stress'],audit=result['audit'],production_promoted=False)
    safe_json=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Final V1 · 빠른 제어 실험 결과</title><style>
    :root{color-scheme:light;--ink:#183344;--muted:#567080;--green:#168472;--line:#d6e1e6}*{box-sizing:border-box}
    body{margin:0;background:#f0f5f6;color:var(--ink);font:15px/1.7 "Malgun Gothic",-apple-system,system-ui,sans-serif}
    main{max-width:1170px;margin:auto;padding:24px}header{background:#153143;color:#f7fbfc;border-radius:18px;padding:30px 34px}
    header small{color:#8ed9ca;letter-spacing:.1em;font-weight:700}h1{font-size:clamp(26px,4vw,38px);line-height:1.3;margin:9px 0 13px}
    header p{color:#d2e3eb;margin:8px 0}a{color:#126ba2;text-underline-offset:4px}header a{color:#aae8da}.nav{display:flex;flex-wrap:wrap;gap:19px;margin-top:18px}
    .card{background:white;border:1px solid var(--line);padding:24px;border-radius:14px;margin-top:16px}h2{font-size:21px;line-height:1.4;margin:0 0 12px}p{margin:10px 0}
    .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.stat{background:white;border:1px solid var(--line);padding:15px 18px;border-radius:12px}.stat strong{display:block;font-size:27px}.stat span,.note{color:var(--muted);font-size:13px}
    .callout{background:#eaf5f0;border-left:4px solid var(--green);padding:13px 17px;border-radius:5px}.warn{background:#faf0e9;border-left-color:#b57644}
    .table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{padding:11px 10px;border-bottom:1px solid #dfe7eb;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#f1f6f7}tr.candidate{background:#e9f5ef;font-weight:700}
    img{width:100%;height:auto;display:block;margin-top:15px;border:1px solid #e4eaee;border-radius:10px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.twocol .card{margin-top:0}button{background:var(--green);color:white;border:0;border-radius:8px;padding:10px 15px;font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #5698b4;outline-offset:3px}details summary{cursor:pointer;font-weight:700}footer{padding:24px 5px;color:var(--muted);font-size:12px}
    @media(max-width:650px){main{padding:12px}header{padding:23px 21px}.card{padding:17px}.stats{grid-template-columns:1fr 1fr;gap:8px}.stat{padding:12px}.stat strong{font-size:24px}.twocol{grid-template-columns:1fr}h2{font-size:19px}}
    </style></head><body><main>
    <header><small>FINAL V1 · CONTROL CADENCE STUDY</small><h1>빠른 명령 적용 주기 실험 결과</h1>
    <p>다음 전체 임무 시험 후보는 <b>0.5초 판단 + 1Hz 관측</b>입니다. 이번 조건에서 새 단절은 9.7%, 단절 시간은 5.6% 줄었습니다.</p>
    <p><b>이 수치는 기존 Final V1 전체 임무 성능과 구분합니다.</b> 같은 출발 위치에서 새 운동·통신 조건으로 실행한 24초 분기 실험입니다.</p>
    <div class="nav"><a href="./">비행·재배치 재생</a><a href="history.html">단절 전 30초 분석</a><a href="#comparison">전체 결과표</a></div></header>
    <div class="stats"><div class="stat"><strong>1,016회</strong><span>주 비교 904회 + 지연 시험 112회</span></div><div class="stat"><strong>113조건</strong><span>기존 42비행 · 14개 지도에서 출발</span></div><div class="stat"><strong>0건</strong><span>안전 위반 · 기존 모델 보존</span></div><div class="stat"><strong>0.1초</strong><span>새 전파·패킷·운동 계산 간격</span></div></div>
    <section class="card" id="comparison"><h2>주기를 줄였을 때 얻은 것과 비용</h2>
    <div class="callout"><b>3초 → 0.5초 판단</b>에서 새 단절 62→56건, 단절 시간 294.5→278.0초, 기한 내 전달률 72.87→73.37%였습니다.</div>
    <p>모든 기체를 10Hz로 보고하게 한 경우에는 단절이 추가로 줄지 않았고 전달률은 52.75%로 낮아졌습니다.
    중요 링크만 빠르게 보고한 경우도 전달률 부담이 남았습니다. 비컨과 명령의 실제 점유 시간을 동일한 데이터 채널에서 차감한 결과입니다.</p>
    <div class="table-wrap"><table><thead><tr><th>구성</th><th>새 단절</th><th>단절 시간(초)</th><th>기한 내 전달</th><th>관측·명령 점유</th><th>이동 방향 반전</th></tr></thead><tbody>__ROWS__</tbody></table></div>
    <p class="note">점유율은 기체당 비컨 5ms와 100ms마다 명령 전달 세 슬롯을 쓰는 고정 TDMA 가정입니다. 실제 장비의 통신 비용을 측정한 값이 아닙니다.
    방향 반전은 이동 방향이 직전 진행 방향과 120도 이상 벌어진 횟수입니다.</p>
    <img src="__COMPARISON_IMAGE__" alt="여덟 구성의 단절 수·단절 시간·전달률·중계 이동량 비교 그래프">
    <p><button id="download">결과표 CSV 내려받기</button></p></section>
    <section class="card"><h2>차이가 어느 정도 확실한가</h2>
    <p>0.5초 판단의 새 단절 차이는 <b>−6건, 지도 단위 재표집 95% 구간 −12~−1건</b>입니다.
    단절 시간 차이는 <b>−16.5초, 95% 구간 −35.2~−1.5초</b>였습니다.
    중계 이동량은 약 0.20% 늘었고 방향 반전은 94→100회였습니다.</p>
    <p>113조건 중 단절 수가 줄어든 조건은 7개, 늘어난 조건은 1개, 같은 조건은 105개입니다.
    선택된 위험 구간의 결과이므로 새 지도 일반화나 전체 임무의 오경보/h를 뜻하지 않습니다.</p>
    <div class="table-wrap"><table><thead><tr><th>3초 판단 대비</th><th>새 단절 차이</th><th>지도 재표집 95% 구간</th><th>전달률 차이</th></tr></thead><tbody>__CI__</tbody></table></div></section>
    __EXAMPLE__
    <div class="twocol"><section class="card"><h2>실기체 적용을 위해 남은 점</h2>
    <p>감시를 빠르게 해도 관측 전달, 계산 완료, 명령 수신, 실제 이동의 시간이 각각 필요합니다.</p>
    <p>이번에는 센서 처리 0.1초, 계획 계산 0.2초, 명령 전달 기본 0.1초를 반영했습니다. 계획기가 바쁠 때 겹친 요청은 건너뛰었습니다.</p>
    <p>최대 속도 5m/s, 가속도 2.5m/s², 최소 분리 5m를 지켰습니다. 운동 모형은 안전 간선의 끝에서 정지한 뒤 다음 방향을 선택합니다.
    간선 중간의 긴급 제동·선회는 이번 모형에 포함하지 않았습니다.</p></section>
    <section class="card"><h2>채택 판단</h2><div class="callout"><b>1초·0.5초 판단과 기존 1Hz 관측</b>을 다음 전체 임무 검증 후보로 남겼습니다. 운영 Final V1은 교체하지 않았습니다.</div>
    <p>0.1초 감시 자체의 추가 효과는 작았습니다. 전 기체의 보고 주기를 높이는 방식은 데이터 전달 비용 때문에 채택 기준을 통과하지 못했습니다.</p>
    <p>후속 검증에서는 비컨을 늘리기 전에 기존 패킷에 관측을 함께 싣는 방법, 중요한 기체만 짧게 보고하는 방법,
    실제 통신 모듈의 지연과 처리량을 확인해야 합니다.</p></section></div>
    <section class="card"><h2>명령 전달이 0.3초 걸린 추가 시험</h2><p>지도별 한 조건, 총 14조건에서 전달 지연을 늘렸습니다. 모든 구성의 새 단절은 7건으로,
    이 별도 표본에서는 빠른 판단의 추가 예방 효과를 확인하지 못했습니다. 주 비교 113조건과 표본 수가 다릅니다.</p>
    <div class="table-wrap"><table><thead><tr><th>구성</th><th>조건</th><th>새 단절</th><th>단절 시간(초)</th><th>기한 내 전달</th></tr></thead><tbody>__STRESS__</tbody></table></div></section>
    <details class="card"><summary>시험 범위와 검증 근거</summary><p>기존 V21 비행 42개에서 위험 구간 71곳과 연결 유지 구간 42곳을 고정 규칙으로 골랐습니다.
    모든 방식은 같은 위치·정지 상태·빈 패킷 큐에서 시작합니다. 같은 물리 위험 판단을 서로 다른 주기로 실행했습니다.
    PPO나 LightGBM을 새로 학습한 시험이 아니며, 원본 V21 정책의 전체 임무를 그대로 고주기로 바꾼 성능도 아닙니다.</p>
    <p>원본 항법 생성기를 출발 시점까지 재현한 뒤, 당시 발행한 보정을 유지하는 미래 계획을 사용했습니다.
    이후 원본에 기록된 실제 행동·위치·전파를 예측 입력으로 주지 않았습니다. 공유 항법 상태와 발행 계획은 정확하다는 가정입니다.</p>
    <p>0.1초 전파는 위치에서 새로 계산했습니다. 그림자 변화는 기존 0.5초 끝점에 조건부인 새 OU 과정입니다.
    기존 두 점을 직선으로 나눈 기록이나 실기체 관측이 아닙니다. 성공적으로 수신한 과거 RF만 사용했습니다.</p>
    <p>1,016개 저장 기록과 243,840개 전파 프레임을 재집계했습니다. 위치·속도·가속도·안전거리·패킷 보존·명령 지연·기존 코드 보존을 확인했습니다.
    개발 시험의 통신 비용에 따라 추가한 두 중요 링크 보고 방식도, 새 확인 결과를 열기 전에 별도로 등록했습니다.</p></details>
    <footer>Final V1 · 완료 2026-11-05 17:37 KST · 본인 전용 결과 요약 · 실기체 검증 및 운영 교체는 미실시</footer>
    </main><script type="application/json" id="summary-data">__SAFE_DATA__</script><script>
    const data=JSON.parse(document.querySelector('#summary-data').textContent);
    document.querySelector('#download').addEventListener('click',()=>{
      const rows=['configuration,persistent_outages,persistent_seconds,timely_pdr,control_airtime_fraction,relay_travel_m,reversals'];
      for(const [key,v] of Object.entries(data.summary))rows.push([key,v.onsets,v.cut_seconds.toFixed(3),v.timely_pdr,v.airtime_fraction,v.relay_travel_m.toFixed(3),v.reversals].join(','));
      const blob=new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='Final_V1_fast_control_summary.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });</script></body></html>'''
    page=template.replace('__ROWS__',''.join(rows)).replace('__CI__',''.join(ci)).replace('__STRESS__',''.join(stress)).replace('__EXAMPLE__',sample_html).replace('__COMPARISON_IMAGE__',image(study/'빠른제어_비교.png')).replace('__SAFE_DATA__',safe_json)
    target=out/'fast-control.html';target.write_text(page,encoding='utf-8')
    # Enforce the advertised upload boundary before asking the publisher to act.
    for forbidden in ('C:\\Users','C:/Users','archive_path','source_path','source_repository_credential','Bearer '):
        assert forbidden not in page,forbidden
    manifest=dict(file=str(target),sha256=digest(target),size_bytes=target.stat().st_size,
        contents='Aggregate metrics, two rendered scientific figures, method/limitations text; '
                 'no source code, model weights, raw packet logs, coordinates, file paths or credentials.',
        destination_project_id='appgprj_6aec141784108191b49e92fd7ab851ac',
        destination_url='https://capstone-final-v1-replay-1105.dododom713.chatgpt.site/fast-control.html',
        expected_access='Existing owner-private site, owner role verified; one allowed user and no groups.',
        source_result_sha256=digest(art/'results.json'),study_completed=done['state']=='completed')
    (out/'publication_payload.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
