"""Only declared observed features and frozen neural context cross inference."""
import numpy as np
from .contracts import sha

def load(item):
    if sha(item['path'])!=item['sha256'] or sha(item['parent_path'])!=item['parent_sha256']:raise ValueError('Input binding changed')
    with np.load(item['path'],allow_pickle=False) as a:out={k:a[k] for k in ('base','network','physics','network_history','ends','n','ns','baseline')}
    with np.load(item['parent_path'],allow_pickle=False) as a:out.update({k:a[k] for k in ('embedding','logits','history')})
    return out
