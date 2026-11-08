"""Matched development comparison; frozen campaign and final test are untouched."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sys
import time

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
from adaptive_joint import ARTIFACTS
from adaptive_joint.contracts import read_json, atomic_json, digest, sha
from adaptive_joint.experiment import assert_protocol, collect, validation_specs
from adaptive_joint.driver import read_episodes
from adaptive_joint.metrics import summarize, paired_interval
from adaptive_joint.replay import compare

FOLDER = ARTIFACTS / 'comparison_v5_v6_20261031'
RUN = ARTIFACTS / 'runs/adaptive_v6_20261030'


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def key(row):
    s = row.get('spec', row)
    return tuple(s[k] for k in ('region', 'direction', 'kind', 'seed'))


def status(stage, **fields):
    item = dict(stage=stage, updated_at=now(), **fields)
    atomic_json(FOLDER / 'status.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def main():
    started = time.perf_counter()
    protocol = read_json(ARTIFACTS / 'protocol.json')
    assert_protocol(protocol)
    specs = validation_specs(8, 'selection')
    checkpoint = RUN / 'checkpoints/s0000144.pt'
    assert sha(checkpoint) == '735909206e4c701db8ead31455cdbae9a34b4daaf0c0fa4904da9b8e30f2868a'
    expected = {key(s) for s in specs}
    new_rows = {}
    for step in (96, 144):
        rows = [read_json(p) for p in sorted((RUN / 'validation' / f'u{step:07d}' / 'rollouts').glob('*.json'))]
        assert len(rows) == 8 and {key(r) for r in rows} == expected
        assert all(r['total_drones'] == 8 and r['search_drones'] == 3 and r['deployment'] == 'v8_s3_r5_40' for r in rows)
        new_rows[step] = {key(r): r for r in rows}

    status('legacy_full_missions', cases=len(specs), prediction_device='cpu', training_unchanged=True)
    old = collect(checkpoint, protocol, specs, FOLDER / 'legacy_v5', 'legacy_v5', True, load_arrays=False)
    old_summary = summarize(read_episodes(old))
    communication = dict(old176=old_summary, comparisons={}, new={})
    keys = ['search_connected_mean', 'pdr', 'outage_total_s', 'outage_max_s', 'duration_s', 'delay_p95_s', 'relay_travel_m']
    for step in (96, 144):
        summary = read_json(RUN / 'validation' / f'u{step:07d}' / 'summary.json')
        communication['new'][str(step)] = summary
        pairs = [(r, new_rows[step][key(r)]) for r in old]
        communication['comparisons'][str(step)] = {m: paired_interval(pairs, m) for m in keys}
    communication.update(scope='Same 8 development cases, same 8-agent V40 fleet, paired environment; not independent test',
                         protocol_sha256=digest(protocol), evaluated_at=now(),
                         packet_rng_note='Absolute jammer and shared shadow traces are matched; packet random draws follow actual actions')
    atomic_json(FOLDER / 'communication.json', communication)
    status('matched_prediction_replay', communication_ready=True, prediction_device='cpu', cases=8)
    prediction = compare(checkpoint, protocol, FOLDER / 'legacy_v5', device='cpu')
    atomic_json(FOLDER / 'prediction.json', prediction)
    assert_protocol(protocol)
    assert not (ARTIFACTS / 'test_access.json').exists()

    percent = lambda x: f'{x*100:.2f}%' if x is not None else '미정'
    communication_rows = []
    for title, field, fmt in [
        ('CTR: 수색 드론 간 전체 연결 시간', 'search_connected_mean', percent),
        ('PDR: 패킷 전달률', 'pdr', percent),
        ('누적 단절 시간', 'outage_total_s', lambda x: f'{x:.2f}초'),
        ('개별 비행 최장 단절의 평균', 'outage_max_s', lambda x: f'{x:.2f}초'),
        ('임무 시간', 'duration_s', lambda x: f'{x:.2f}초'),
        ('개별 비행 p95 지연의 평균', 'delay_p95_s', lambda x: f'{x*1000:.2f}ms'),
        ('중계 이동 거리', 'relay_travel_m', lambda x: f'{x/1000:.2f}km')]:
        values = [old_summary['means'][field]] + [communication['new'][str(s)]['means'][field] for s in (96, 144)]
        communication_rows.append('| '+title+' | '+' | '.join(fmt(x) for x in values)+' |')

    prediction_rows = []
    for title, target, field in [
        ('5초 내 단절 AP', 'next_onset_lt5s', 'average_precision'),
        ('5초 내 단절 AUROC', 'next_onset_lt5s', 'auroc'),
        ('5초 내 단절 표본 정밀도', 'next_onset_lt5s', 'precision'),
        ('5초 내 단절 표본 재현율', 'next_onset_lt5s', 'recall'),
        ('3~4초 단절 AP', 'next_onset_3to4s', 'average_precision'),
        ('3~7초 단절 AP', 'next_onset_3to7s', 'average_precision')]:
        values = [prediction['models'][m][target][field] for m in ('old176_scores_on_v6_targets', 'warm_1s5500', 'joint_v6')]
        prediction_rows.append('| '+title+' | '+' | '.join(percent(x) for x in values)+' |')

    report = f'''# 캡스톤 디자인 II — 구 176 모델과 신규 공동학습 직접 비교

평가 완료: {now()}. 본학습은 별도로 계속 진행. 비교 중 모델 소스·학습 설정 변경 없음.

## 통신·임무 성능: 같은 8개 시나리오, 같은 기체 구성

모든 모델에 동일한 지도·방향·간섭 유형·난수 조건 사용. 전체 8 / 수색 3 / 중계 5대, V형 40m로 자원 동일.
구 176은 동결 가중치로 신규 환경에서 전체 임무 재실행. 신규 96·144는 같은 시나리오의 저장 평가 결과 사용.
각 정책이 선택한 이동에 따라 통신 관측과 임무 시간은 달라짐. 지표는 비행별 값의 평균.

| 지표 | 구 176 | 신규 96: 통신 best | 신규 144: 예측 AP best |
|---|---:|---:|---:|
{chr(10).join(communication_rows)}

지도 단위 짝지은 차이와 탐색적 95% 구간은 `communication.json`에 기록.
구 176의 기존 HTML이나 과거 별도 시나리오 수치를 가져온 비교가 아님.

## 예측 성능: 동일 관측·동일 정답

구 176의 위 8개 비행 기록을 고정하고 세 예측기를 재생. 동일한 RF 이력·현재 수신 마스크·기하 입력·예측 시점·지원 범위·정답 사용.
공통 링크·시점 표본: {prediction['common_samples']:,}개. 1초 간격 예측. 문턱 0.5. AP·AUROC는 정답률이 아님.

| 지표 | 구 176 | 1초 사전학습 5,500 | 신규 공동학습 144 |
|---|---:|---:|---:|
{chr(10).join(prediction_rows)}

공통 정답은 v6의 다음 1초 이상 지속 단절 시작이다. 구 176 점수도 이 정답에 대해 다시 평가했다.
따라서 구 176의 원래 any-event 정답으로 계산한 HTML AP와는 다르다.
이 표의 표본 재현율은 실제 사건의 3초 선행 경보 재현율과 구분해야 한다.

## 범위 및 제한

- 사용 지역 8개는 모델 선택에 사용된 개발 지역. 독립 최종 성능이나 전 지도 일반화의 증명이 아님.
- 비교 가중치는 구 176, 신규 통신 best 96, 신규 정기 평가 중 AP 최고 144로 고정. 다른 신규 시점은 이 비교에 포함하지 않음.
- 예측 비교에서는 관측을 고정했으므로 정책 변화로 관측이 달라지는 혼동을 줄였다. 미래 정답은 모델 입력으로 전달하지 않음.
- 32개 최종 평가 지도는 개봉하지 않음.
- `communication.json`, `prediction.json`에 해시·공통 표본·짝지은 비교 원자료 보존.
'''
    (WORKSPACE / 'adaptive_joint/COMPARISON_V5_V6_20261031.md').write_text(report, encoding='utf-8')
    status('completed', communication_ready=True, prediction_ready=True,
           common_samples=prediction['common_samples'], elapsed_s=time.perf_counter()-started,
           report=str(WORKSPACE / 'adaptive_joint/COMPARISON_V5_V6_20261031.md'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        raise
