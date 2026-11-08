"""Read-only frozen-predecessor composition and common V11 model architecture."""
import numpy as np
from .contracts import read,PARENT
from forecast_graph.inference import predict,pin


def parent_choices():
    frozen=read(PARENT/'freeze.json');result={}
    for task,decision in frozen['decisions'].items():
        selected=next(i for i in frozen['candidates'] if i['candidate']==decision['candidate'])
        champion=next(i for i in frozen['candidates'] if i['candidate']['task']==task and i['candidate']['tag']==decision['ranking_champion'])
        result[task]=dict(selected=selected,champion=champion)
    return result


def parent_predict(candidate,arrays,record,ep,cache=None):
    from forecast_graph.evaluate import previous
    cache={} if cache is None else cache;task=candidate['task'];n=ep.values.shape[2]
    if candidate['family']=='previous':
        return previous(task,dict(source_record=record),arrays,ep,'confirmation')
    if candidate['family']=='blend':
        return sum(parent_predict(m,arrays,record,ep,cache)*.5 for m in candidate['members'])
    model=candidate['model']
    if model not in cache:cache[model]=predict(model,arrays,record,ep,city=getattr(ep,'city',None),device='cuda')
    return cache[model]['link'][:,:n] if task=='link' else cache[model]['service']

