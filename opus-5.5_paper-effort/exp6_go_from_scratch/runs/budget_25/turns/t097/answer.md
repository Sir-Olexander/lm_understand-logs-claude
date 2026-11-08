```python
import json, numpy as np, collections
d=json.load(open('/work/pass_15/games.json'))
B=np.load('/work/pass_15/boards.npy')
gs=[g for g in d if g['opponent']=='strong']
print(collections.Counter((g['end'][:3],g['result']) for g in gs))
print(sorted((g['margin'],g['end'][:3]) for g in gs if g['result']==-1))
for gi,g in enumerate(d):
    if g['end']!='repetition' or g['result']!=-1: continue
    n=len(g['moves']); last=B[gi,n-1]
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    print(gi,g['opponent'],g['margin'],'len',n-1-match,[(m['by'][0],m['move'],round(dict(m.get('top5',[])).get(m['move'],0),2)) for m in g['moves'][match+1:]],caps)
```