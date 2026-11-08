```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
for gi,g in enumerate(d):
    if g['end']!='repetition' or g['result']!=-1: continue
    n=len(g['moves'])
    last=B[gi,n-1]
    # find matching earlier board
    match=[k for k in range(n-2) if (B[gi,k]==last).all()]
    seq=[(m['by'][0],m['move']) for m in g['moves'][match[-1]:]] if match else None
    # captures in the cycle
    caps=[]
    for k in range(match[-1]+1,n) if match else []:
        prev=B[gi,k-1]; cur=B[gi,k]
        removed=int(((prev!=0)&(cur==0)).sum())
        caps.append(removed)
    print(gi,g['opponent'],g['margin'],'cycle from ply',match[-1] if match else None,'len',n-1-match[-1] if match else None,'seq',seq,'removed',caps)
```