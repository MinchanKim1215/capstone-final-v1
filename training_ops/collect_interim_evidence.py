"""Combine the prior PDF snapshot, published legacy HTML, and full validation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from html.parser import HTMLParser
import copy
import hashlib
import json
import re
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'drone_forecast_joint_training'
OUT=ROOT/'output/pdf/interim_evidence'
OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'output/pdf/evidence/snapshot.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.table=[]
        elif tag=='tr' and self.table is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(re.sub(r'\s+',' ',''.join(self.cell)).strip());self.cell=None
        elif tag=='tr' and self.row is not None:
            self.table.append(self.row);self.row=None
        elif tag=='table' and self.table is not None:
            self.tables.append(self.table);self.table=None


def main():
    d=read(OLD)
    shutil.copyfile(OLD,OUT/'previous_pdf_snapshot.json')
    frozen=OUT/'previous_pdf_snapshot.json'
    for sid,field in [('D06','best_summary'),('D09','physics_summary')]:
        d['sources'][sid]=dict(path=str(frozen),field=field,sha256=sha(frozen.read_bytes()),
                               meaning='Preserved eight-case summary; live file later expanded to 96 cases.')
    html_path=Path('C:/Users/mckim/Downloads/Transformer_운용개선_결과.html')
    raw=html_path.read_bytes();html=raw.decode('utf-8-sig')
    parser=Tables();parser.feed(html)
    overall=next(table for table in parser.tables if table and table[0]==['항목','기존 입력 Transformer','선정 Transformer'])
    high_confidence=next(table for table in parser.tables if table and table[0]==['항목','이전 보수 점수 기준','기본 고신뢰 알림'])
    conditions=next(table for table in parser.tables if table and any('지역/기록' in cell for row in table for cell in row))
    overall_rows={row[0]:row[1:] for row in overall[1:]}
    confidence_rows={row[0]:row[1:] for row in high_confidence[1:]}
    legacy=dict(source=str(html_path),source_sha256=sha(raw),report_title='Transformer 예측 개선 결과',
                all_tables=parser.tables,overall=overall_rows,high_confidence=confidence_rows,conditions=conditions,
                ap=float(overall_rows['전체 평가 AP'][1]),brier=float(overall_rows['Brier 점수 · 작을수록 유리'][1]),
                samples=int(re.sub(r'\D','',overall_rows['전체 평가 창'][1])),
                regional_macro_ap=float(overall_rows['지역별 AP 단순 평균'][1]),
                high_precision=float(confidence_rows['첫 알림 정밀도'][1].strip('%'))/100,
                high_event_recall=float(confidence_rows['3초 이상 선행 첫 알림으로 포착한 새 단절'][1].strip('%'))/100,
                high_false_alerts=float(confidence_rows['드론·시간당 오경보/늦은 첫 알림'][1]),
                warning='Published rounded HTML figures; old model weights were not rerun, per user preference.')
    assert legacy['samples']==4866832 and legacy['ap']==.685
    assert np.isclose(legacy['high_precision'],.909,rtol=0,atol=1e-12)
    assert np.isclose(legacy['high_event_recall'],.0039,rtol=0,atol=1e-12)
    d['legacy_html']=legacy
    shutil.copyfile(html_path,OUT/'legacy_report_source.html')
    d['discovery']=read(ROOT/'training_ops/baseline_usable_20261025/discovery.json')
    d['model_structure']=read(ROOT/'training_ops/baseline_usable_20261025/model_structure.json')
    d['sources']['U01']=dict(path=str(html_path),sha256=sha(raw))
    for sid,relative in [('U02','training_ops/baseline_usable_20261025/discovery.json'),('U03','training_ops/baseline_usable_20261025/model_structure.json')]:
        path=ROOT/relative;d['sources'][sid]=dict(path=str(path),sha256=sha(path.read_bytes()))
    d['latest']={}
    for name,code in [('joint','c1a9debd37963fcd'),('physics','30c8224c85fc74c0')]:
        path=PROJECT/'results/transformer/evaluation/validation'/code/'summary.json'
        summary=read(path)
        if summary['cases']!=96 or summary['communication']['complete']!=96:
            raise ValueError('Full validation is not complete')
        rows=[];pins=[]
        for prefix in summary['raw_paths']:
            meta=(PROJECT/prefix).with_suffix('.json');source=meta.read_bytes()
            rows.append(json.loads(source));pins.append(dict(path=str(meta),sha256=sha(source)))
        fields=['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s','duration_s','duration_ratio',
                'delay_mean_s','delay_p95_s','aoi_mean_s','relay_travel_m','outage_events']
        means={key:float(np.mean([row[key] for row in rows])) for key in fields}
        for key,value in summary['communication']['means'].items():
            if key in means:
                assert np.isclose(means[key],value,rtol=0,atol=1e-12),(name,key)
        d['latest'][name]=dict(summary=summary,means=means,rows=rows,
                              worst_outage=max(rows,key=lambda row:row['outage_max_s']))
        d['sources']['L01' if name=='joint' else 'L02']=dict(path=str(path),sha256=sha(path.read_bytes()),metadata=pins)
    key=lambda row:tuple(row[k] for k in ['region','direction','kind','seed'])
    before={key(row):row for row in d['latest']['physics']['rows']}
    after={key(row):row for row in d['latest']['joint']['rows']}
    assert set(before)==set(after) and len(after)==96
    d['latest']['paired_differences']={field:[after[q][field]-before[q][field] for q in sorted(before)]
                                     for field in ['search_connected_mean','pdr','outage_total_s']}
    for sid,field,relative in [('L03','selection','drone_forecast_joint_training/results/transformer/frozen_selection.json'),
                               ('L04','progress','drone_forecast_joint_training/results/transformer/train/transformer_main/progress.json'),
                               ('L05','main_status','training_ops/status.json'),
                               ('L06','pilot_status','prediction_improvement/runs/next_outage_1s_v1/status.json')]:
        path=ROOT/relative;value=read(path)
        d['latest'][field]=value
        d['sources'][sid]=dict(path=str(path),sha256=sha(path.read_bytes()))
    d['progress']=d['latest']['progress'];d['main_status']=d['latest']['main_status'];d['pilot_status']=d['latest']['pilot_status']
    d['sources']['D05']=copy.deepcopy(d['sources']['L04'])
    d['sources']['D04']=copy.deepcopy(d['sources']['L05'])
    d['sources']['D19']=copy.deepcopy(d['sources']['L06'])
    pilot_results=ROOT/'prediction_improvement/runs/next_outage_1s_v1/results.json'
    if pilot_results.exists():
        d['pilot_results']=read(pilot_results)
        d['sources']['L07']=dict(path=str(pilot_results),sha256=sha(pilot_results.read_bytes()))
    history_path=PROJECT/'results/transformer/train/transformer_main/validation_history.json'
    history=read(history_path)
    d['validation_history']=[{key:row.get(key) for key in ['update','score','feasible','improved']} for row in history]
    d['sources']['D13']=dict(path=str(history_path),sha256=sha(history_path.read_bytes()))
    d['as_of']=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    d['historical_eight_case_as_of']=read(OLD)['as_of']
    full=d['latest']['joint']['summary']
    d['legacy_descriptive_gaps']=dict(ap=full['prediction']['risk_5s']['average_precision']-legacy['ap'],
                                    brier=full['prediction']['risk_5s']['brier']-legacy['brier'],
                                    precision_pp=(full['alarms']['8s']['first_alert_precision']-legacy['high_precision'])*100,
                                    event_recall_pp=(full['alarms']['8s']['first_alert_event_recall_3s']-legacy['high_event_recall'])*100,
                                    false_alerts_per_drone_hour=full['alarms']['8s']['false_alerts_per_drone_hour']-legacy['high_false_alerts'],
                                    interpretation='Arithmetic differences of non-matched published populations; not estimated causal improvements or degradations.')
    assert d['progress']['committed_updates']==256
    assert d['latest']['selection']['validation_cases']==96
    assert d['latest']['selection']['test_used'] is False
    (OUT/'snapshot.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(as_of=d['as_of'],full_cases=full['cases'],current_ap=full['prediction']['risk_5s']['average_precision'],
                         old_ap=legacy['ap'],gap=d['legacy_descriptive_gaps']['ap'],main_state=d['main_status']['state'],
                         pilot_status=d['pilot_status'],worst_outage=d['latest']['joint']['worst_outage']['outage_max_s']),ensure_ascii=False))


if __name__=='__main__':main()
