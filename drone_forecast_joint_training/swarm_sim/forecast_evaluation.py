"""예측 전용 평가. truth는 사후 평가에만 사용하고 신경망 입력으로 전달하지 않는다."""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from .regions import sha256_file


def binary_metrics(y, probability, threshold=0.5):
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=np.float64)
    if y.shape != probability.shape or np.any(~np.isfinite(probability)):
        raise ValueError('정답과 예측의 크기 또는 유한성을 확인하십시오.')
    if np.any((probability < 0) | (probability > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError('정답은 0/1, 예측은 0~1이어야 합니다.')
    y = y.astype(np.uint8)
    if not 0 <= threshold <= 1+1e-6:
        raise ValueError('경고 기준 범위 오류')
    size = len(y)
    if size == 0:
        return {'samples': 0, 'positives': 0, 'positive_fraction': None,
                **{k:None for k in ('average_precision','auroc','brier','log_loss',
                    'accuracy','precision','recall','f1','false_positive_rate')},
                'tp':0, 'fp':0, 'tn':0, 'fn':0}
    pred = probability >= threshold
    truth = y.astype(bool)
    tp, fp = int(np.sum(pred & truth)), int(np.sum(pred & ~truth))
    tn, fn = int(np.sum(~pred & ~truth)), int(np.sum(~pred & truth))
    two_classes = 0 < int(y.sum()) < size
    p = np.clip(probability, 1e-7, 1-1e-7)
    return {'samples':size, 'positives':int(y.sum()), 'positive_fraction':float(y.mean()),
            'average_precision':float(average_precision_score(y, probability)) if two_classes else None,
            'auroc':float(roc_auc_score(y, probability)) if two_classes else None,
            'brier':float(np.mean((probability-y)**2)),
            'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'accuracy':(tp+tn)/size, 'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,
            'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}


def validation_threshold(y, probability, currently_available, split='val'):
    """검증셋의 현재 가용 링크에서 F1 최대화. 동률이면 경고가 적은 높은 기준 선택."""
    if split != 'val':
        raise ValueError('경고 기준은 val에서만 정할 수 있습니다.')
    take = np.asarray(currently_available, bool)
    y, p = np.asarray(y)[take], np.asarray(probability)[take]
    if not len(y) or not 0 < y.sum() < len(y):
        raise ValueError('검증셋의 현재 가용 링크에 양성·음성이 모두 필요합니다.')
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def calibration_bins(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bucket = np.minimum((p*bins).astype(int), bins-1)
    return [{'lower':i/bins, 'upper':(i+1)/bins, 'samples':int(np.sum(bucket == i)),
             'mean_risk':float(p[bucket == i].mean()) if np.any(bucket == i) else None,
             'actual_fraction':float(y[bucket == i].mean()) if np.any(bucket == i) else None}
            for i in range(bins)]


@dataclass
class EvaluationContext:
    current_available: np.ndarray
    events: list
    regions: np.ndarray
    episode_ids: np.ndarray


def evaluation_context(dataset, positions):
    """현재 가용성에 따른 진단 집단과 지속 단절 사건의 사후 참조값을 만든다.

현재 가용성은 평가 집단을 나누는 oracle 정보이며 배포 시 알 수 있는 입력이 아니다.
사건은 데이터셋에 한 번 이상 나타난 방향별 링크의 지속 단절 시작으로 정의한다.
"""
    if dataset.split not in ('val', 'test'):
        raise ValueError('truth를 읽는 사후 평가는 val/test에 한정합니다.')
    index = dataset.index[positions]
    current = np.zeros(len(index), bool)
    regions = np.empty(len(index), dtype=object)
    ids = np.empty(len(index), dtype=object)
    events = []
    dt = float(dataset.meta['radio_dt_s'])
    persistence = int(dataset.meta['minimum_outage_steps'])
    for epno in np.unique(index[:, 0]):
        ep = dataset.collection['episodes'][int(epno)]
        path = dataset.collection_folder / ep['path'] / 'truth.npz'
        if sha256_file(path) != ep['files_sha256']['truth.npz']:
            raise ValueError('평가 진실값 파일의 해시가 다릅니다.')
        with np.load(path, allow_pickle=False) as truth:
            available, times = truth['link_available'], truth['time_end_s']
        rows = np.flatnonzero(index[:, 0] == epno)
        k, src, dst = index[rows, 1:].T
        current[rows] = available[k, src, dst]
        regions[rows], ids[rows] = ep['region'], ep['id']
        for source, destination in np.unique(index[rows, 2:4], axis=0):
            link_rows = rows[(index[rows, 2] == source) & (index[rows, 3] == destination)]
            anchors = index[link_rows, 1]
            bad = ~available[:, source, destination]
            boundary = np.diff(np.r_[False, bad, False].astype(np.int8))
            starts, stops = np.flatnonzero(boundary == 1), np.flatnonzero(boundary == -1)
            for start, stop in zip(starts, stops):
                if stop-start < persistence:
                    continue
                # 정답과 동일하게 지속 구간 전체가 미래 창 안에 들어와야 한다.
                eligible = ((anchors < start) & (start+persistence-1 <= anchors+dataset.horizon)
                            & current[link_rows])
                candidates = link_rows[eligible]
                events.append({'region':ep['region'], 'episode_id':ep['id'],
                    'source':int(source), 'destination':int(destination),
                    'start_s':float(times[start]-dt), 'duration_s':float((stop-start)*dt),
                    'candidate_positions':candidates,
                    'candidate_times_s':times[index[candidates, 1]]})
    return EvaluationContext(current, events, regions, ids)


def event_metrics(context, probability, threshold):
    rows = []
    for event in context.events:
        positions = event['candidate_positions']
        alerted = probability[positions] >= threshold
        warning = float(event['candidate_times_s'][alerted].min()) if np.any(alerted) else None
        lead = max(0., event['start_s']-warning) if warning is not None else None
        rows.append({k:v for k,v in event.items() if not k.startswith('candidate_')} |
                    {'eligible':bool(len(positions)), 'detected':warning is not None,
                     'first_warning_s':warning, 'lead_time_s':lead,
                     'strictly_early':lead is not None and lead > 1e-8})
    eligible = [r for r in rows if r['eligible']]
    detected = [r for r in eligible if r['detected']]
    leads = [r['lead_time_s'] for r in detected]
    summary = {'definition':'관측 창이 존재한 방향별 링크의 지속 단절. 현재 가용한 과거 창에서 평가.',
        'all_outage_events_in_observed_links':len(rows), 'eligible_events':len(eligible),
        'excluded_events_without_eligible_window':len(rows)-len(eligible),
        'detected_events':len(detected),
        'event_recall':len(detected)/len(eligible) if eligible else None,
        'strictly_early_recall':sum(r['strictly_early'] for r in eligible)/len(eligible) if eligible else None,
        'lead_time_median_s':float(np.median(leads)) if leads else None,
        'lead_time_min_s':float(np.min(leads)) if leads else None,
        'lead_time_p10_s':float(np.percentile(leads, 10)) if leads else None,
        'lead_time_max_s':float(np.max(leads)) if leads else None,
        'scope':'현재 가용성은 사후 평가에만 사용. 사건 없음/관측 부족을 성공으로 계산하지 않음.'}
    return summary, rows


def grouped_metrics(y, probability, context, threshold):
    result = {'all':binary_metrics(y, probability, threshold),
              'currently_available':binary_metrics(y[context.current_available],
                                    probability[context.current_available], threshold),
              'by_region':{}, 'by_episode':{}}
    for name, labels in [('by_region', context.regions), ('by_episode', context.episode_ids)]:
        for key in sorted(set(labels)):
            take = labels == key
            up = take & context.current_available
            result[name][str(key)] = {'all':binary_metrics(y[take], probability[take], threshold),
                                    'currently_available':binary_metrics(y[up], probability[up], threshold)}
    valid = [r['currently_available']['average_precision'] for r in result['by_region'].values()
             if r['currently_available']['average_precision'] is not None]
    result['macro_available_region_ap'] = float(np.mean(valid)) if valid else None
    result['macro_available_region_ap_regions'] = len(valid)
    result['macro_available_region_ap_excluded_regions'] = len(result['by_region'])-len(valid)
    result['calibration'] = calibration_bins(y, probability)
    return result
