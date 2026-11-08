"""New study boundaries; V9 artifacts and the 32 final maps remain read-only."""
from . import ROOT,ARTIFACTS,VERSION
from forecast_extension import ARTIFACTS as V9
from forecast_extension.contracts import read,write,sha,digest,now,maps,folds,role,BASELINE
from forecast_extension.evaluate import verify_freeze

SEED=2026110203


def prepare():
    verify_freeze()
    value=dict(version=VERSION,seed=SEED,created_at=now(),parent_freeze_sha256=sha(V9/'freeze.json'),
        parent_delivery_sha256=sha(V9/'delivery_manifest.json'),parent_candidate_sha256=sha(V9/'research_candidate.json'),
        stages=['Selection-only event diagnostics','Packet-service target and causal-input construction','One-change-at-a-time training ablations','Freeze then fresh-flight confirmation and audit'],
        diagnosis_scope='V9 selection flights only; V9 confirmation is never used for optimization',
        original_final_test_opened=False,production_ready=False,hardware_validated=False,
        notes=['Causal inputs remain received past telemetry, received issued commands, and public GIS/nominal plans.',
               'True future actions, packet outcomes, radio state and jammer schedules are diagnosis/labels only.',
               'Current packet engine is ideal TDMA, not a measured radio or IP/Wi-Fi stack.',
               'Diagnostic association flags do not establish causes.'])
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        for key in value:
            if key!='created_at' and old[key]!=value[key]:raise ValueError('V10 study boundary changed')
        return old
    write(path,value);return value


def verify():
    p=read(ARTIFACTS/'protocol.json');verify_freeze()
    for name,key in [('freeze.json','parent_freeze_sha256'),('delivery_manifest.json','parent_delivery_sha256'),('research_candidate.json','parent_candidate_sha256')]:
        if sha(V9/name)!=p[key]:raise ValueError('Prior study changed: '+name)
    return p


def status(stage,**kwargs):write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),**kwargs))
