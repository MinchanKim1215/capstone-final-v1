"""Standalone comparison with explicit exploratory screening criteria."""
# 현재 비교 HTML 생성기입니다. 비교 화면 틀과 실험 JSON을 결합하며 시뮬레이터를 실행하지 않습니다.
# 실제 화면·탐색 판정·정렬 코드는 옆의 comparison_screening.html에 있습니다.
from pathlib import Path
import json

HTML = Path(__file__).with_name("comparison_screening.html").read_text(encoding="utf-8")

# 저장된 집계 JSON을 오프라인 HTML에 삽입합니다. </script> 같은 문자열이 태그를 닫지 않도록 이스케이프합니다.
def write_comparison(path, data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    Path(path).write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
