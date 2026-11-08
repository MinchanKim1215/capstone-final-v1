"""전체 원시 기록을 독립 에피소드 단위, 450MiB 이하 ZIP으로 보존."""
import json,zipfile
from pathlib import Path
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from usable.data import collection


def main():
    manifest,p=collection();dest=ROOT.parent/'deliverables';dest.mkdir(exist_ok=True)
    maximum=450*1024**2;groups=[];current=[];size=0
    for r in manifest['records']:
        paths=[ROOT/r['path']/n for n in ('episode.json','observations.npz','truth.npz','spatial_observations.npz')]
        total=sum(x.stat().st_size for x in paths)
        if current and size+total>maximum:groups.append(current);current=[];size=0
        current.append((r,paths));size+=total
    if current:groups.append(current)
    rows=[]
    for q,group in enumerate(groups,1):
        name=f'drone_forecast_data_part{q:02d}.zip';path=dest/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for r,files in group:
                for f in files:z.write(f,'drone_forecast_usable/'+f.relative_to(ROOT).as_posix())
            if q==1:z.write(ROOT/'datasets/usable45/collection_manifest.json','drone_forecast_usable/datasets/usable45/collection_manifest.json')
        row={'file':name,'bytes':path.stat().st_size,'sha256':sha256_file(path),'records':[r['id'] for r,_ in group]};rows.append(row)
        print(name,row['bytes'],len(group),flush=True)
    write_json(ROOT/'results/usable/raw_archive_index.json',{'format':'usable-raw-archives-v1','parts':rows,'total_records':len(manifest['records'])})
    text='# 전체 원시 기록\n\n270개 전체 비행의 기본 관측·지도 파생 관측·정답·메타데이터. 아래 압축 파일을 코드 압축 파일과 같은 폴더에 모두 해제하여 병합. ZIP 내부 최상위 폴더: drone_forecast_usable.\n\n'
    for r in rows:text+='- '+r['file']+' · '+f'{r["bytes"]/1024**2:.1f} MiB'+' · '+str(len(r['records']))+'개 비행\n  SHA-256: `'+r['sha256']+'`\n'
    text+='\n코드 묶음에 포함된 3개 시연 기록만으로 재생 가능. 전체 재학습·평가에는 위 원시 기록 또는 collect 명령의 전체 재수집 필요.\n'
    (ROOT/'DATA_INDEX.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
