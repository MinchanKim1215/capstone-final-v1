"""Prepare lossless calibration input features, without scoring any model."""
from . import ARTIFACTS
from .contracts import read,write,now,verify_protocol
from .feature_cache import CachedEpisode
import numpy as np


def main():
    verify_protocol()
    rows=[r for r in read(ARTIFACTS/'dataset.json')['rows'] if r['role']=='calibration']
    for index,record in enumerate(rows):
        episode=CachedEpisode(record);query=np.argwhere(episode.valid)[:1]
        if len(query):episode.features(query,True,True)
        write(ARTIFACTS/'prepare_static_status.json',dict(state='running',scope='calibration',complete=index+1,total=len(rows),updated_at=now()))
    write(ARTIFACTS/'prepare_static_status.json',dict(state='completed',scope='calibration',complete=len(rows),total=len(rows),
        updated_at=now(),prediction_or_model_selection_performed=False))


if __name__=='__main__':main()
