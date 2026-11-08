"""Render saved full mission coordinates and metrics without altering results."""
# 저장된 전체 임무 좌표와 지표를 재생 HTML에 넣습니다.
# HTML 문자열은 화면 코드이고, 내부 __DATA__ 자리는 실제 기록으로 대체합니다.
# 화면 수정은 이동·채널·성능 수치를 새로 계산하는 작업과 구분합니다.
import json
# 재생 화면의 HTML·JavaScript는 읽기 쉬운 별도 파일에 보관합니다.
from pathlib import Path
HTML = Path(__file__).with_name("replay_full_template.html").read_text(encoding="utf-8")

# 기존 result의 프레임·수치에 지도 표시 정보만 붙여 HTML을 만듭니다. 패킷이나 궤적을 다시 계산하지 않습니다.
def build_full_replay(result,scene):
    data=dict(result)
    data['buildings']=[{'geometry':b['geometry_local_m'],'height':result['effective_heights_m'][i],'imputed':b['height_source']=='imputed'} for i,b in enumerate(scene['buildings'])]
    data['region_name']=scene['metadata']['region_name']
    data['successful_hops']={}
    return HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
