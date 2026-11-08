"""브라우저를 실행하지 않는 게시 HTML·JS·프레임·배포 연결 검수."""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from cooperative.continuity import STUDY, protocol, verify_selection
from cooperative.refine import atomic_json
from swarm_sim.regions import ROOT, sha256_file


class Document(HTMLParser):
    def __init__(self,text):
        super().__init__();self.scripts=[];self.images=[];self.ids=set();self.current=None
        self.feed(text);self.close()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids:raise ValueError('Duplicate HTML id: '+a['id'])
            self.ids.add(a['id'])
        if tag=='script':
            if a.get('src'):raise ValueError('Standalone HTML must not require external scripts')
            self.current=dict(attrs=a,text='')
        if tag=='img':self.images.append(a.get('src',''))
        if tag=='link' and a.get('rel')=='stylesheet':raise ValueError('Standalone report must embed its styles')
    def handle_endtag(self,tag):
        if tag=='script' and self.current is not None:self.scripts.append(self.current);self.current=None
    def handle_data(self,text):
        if self.current is not None:self.current['text']+=text


def verify():
    deployment=(ROOT/'results/deployment/published_selection.json').exists()
    if deployment:
        from cooperative.deployment_publish import verify_publication
        pub=verify_publication();study=ROOT/'results/deployment';freeze={'selected':pub['selected']}
        report=ROOT/'docs/통신우선_최종확인_v3_1.html';replay=ROOT/'docs/통신우선_전체임무_재생_v3_1.html'
    else:
        plan=protocol();freeze=verify_selection(plan);study=STUDY
        pub=json.loads((STUDY/'published_selection.json').read_text(encoding='utf-8'))
        report=ROOT/'docs/통신우선_결과_v3.html';replay=ROOT/'docs/통신우선_전체임무_재생_v3.html'
    if (sha256_file(report)!=pub['report_sha256'] or sha256_file(replay)!=pub['replay_sha256']):
        raise ValueError('Published HTML digest mismatch')
    report_text=report.read_text(encoding='utf-8');replay_text=replay.read_text(encoding='utf-8')
    if any(re.search(r'__[A-Z_]+__',s) for s in [report_text,replay_text]):raise ValueError('Unresolved HTML placeholder')
    doc=Document(report_text);rdoc=Document(replay_text)
    if len(doc.images)!=2 or doc.scripts:raise ValueError('Expected two embedded scientific plots and static report')
    for src in doc.images:
        if not src.startswith('data:image/png;base64,'):raise ValueError('Plot is not self-contained')
        data=base64.b64decode(src.split(',',1)[1],validate=True)
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Embedded scientific plot is invalid PNG')
    payloads=[s for s in rdoc.scripts if s['attrs'].get('id')=='data']
    if len(payloads)!=1:raise ValueError('One embedded replay data payload required')
    payload=json.loads(payloads[0]['text'])
    if len(payload['cases'])!=3 or not payload['validation_selected']:raise ValueError('Fixed three validation-selected cases required')
    ids={'case','play','reset','speed','time','clock','links','risks','paths','observer',
         'map0','map1','timeline','stats0','stats1','finaltable','risktable'}
    if not ids<=rdoc.ids:raise ValueError('Replay control/data targets missing')
    counts=[]
    for case in payload['cases']:
        if len(case['runs'])!=2:raise ValueError('Each replay needs baseline and selected policy')
        for run in case['runs']:
            frames=run['frames'];summary=run['summary']
            n=int(round(summary['duration_s']/3))
            if (len(frames)!=n+1 or not summary['mission_completed'] or frames[0]['time_s']!=0
                or frames[-1]['time_s']!=summary['duration_s']):raise ValueError('Incomplete replay mission')
            if [f['time_s'] for f in frames]!=list(range(0,3*(n+1),3)):raise ValueError('Nonuniform replay time')
            for f in frames:
                pos=np.asarray(f['positions'],float);risk=np.asarray(f['risk'],float);links=np.asarray(f['links'])
                if (pos.shape!=(8,2) or not np.isfinite(pos).all() or np.max(np.abs(pos))>200
                    or risk.shape!=(8,8) or not np.isfinite(risk).all() or np.any((risk<0)&(risk!=-1)) or np.any(risk>1)
                    or links.shape!=(8,8) or not np.isin(links,[0,1]).all() or not np.array_equal(links,links.T)):
                    raise ValueError('Invalid replay frame geometry/risk/links')
            counts.append(dict(region=summary['region'],controller=summary['controller'],frames=len(frames),duration_s=summary['duration_s']))
    node=os.environ.get('CODEX_PRIMARY_RUNTIME_NODE') or shutil.which('node')
    if not node:raise RuntimeError('Node is required for JavaScript syntax verification')
    checked=0
    for script in rdoc.scripts:
        if script['attrs'].get('type')=='application/json':continue
        with tempfile.TemporaryDirectory(prefix='capstone-js-check-') as folder:
            p=Path(folder)/'replay.js';p.write_text(script['text'],encoding='utf-8')
            result=subprocess.run([node,'--check',str(p)],text=True,capture_output=True)
            if result.returncode:raise ValueError('Replay JavaScript syntax failed: '+result.stderr)
            checked+=1
    if checked!=1:raise ValueError('Expected exactly one replay JavaScript program')
    aliases={'docs/통합_결과.html':report,'docs/통합_전체임무_재생.html':replay,'docs/통신위험_전체임무_최종.html':replay}
    for alias,original in aliases.items():
        if sha256_file(ROOT/alias)!=sha256_file(original):raise ValueError('Front-door HTML differs: '+alias)
    result=dict(format='deployment-static-artifact-verification-v3.1' if deployment else 'communications-priority-static-artifact-verification-v3',passed=True,
        report_plots=2,replay_cases=3,replay_full_missions=6,frames=counts,javascript_syntax_programs=checked,
        standalone_runtime_assets=True,report_sha256=sha256_file(report),replay_sha256=sha256_file(replay),
        selected=freeze['selected'],interactive_browser_qa=False,
        limitation='browser file access blocked; no browser or network workaround; static validation only')
    atomic_json(study/'verification/artifact_static_review.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':verify()
