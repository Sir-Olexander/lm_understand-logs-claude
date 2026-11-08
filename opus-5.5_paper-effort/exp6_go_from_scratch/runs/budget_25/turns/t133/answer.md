```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_23/games.json')); B=np.load('/work/pass_23/boards.npy')
for gi,g in enumerate(d):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    seq=[(m['by'][0],divmod(m['move'],9),round(dict(m.get('top5',[])).get(m['move'],0),2)) for m in g['moves'][match+1:]]
    print(gi,g['opponent'][0],g['result'],g['margin'],'len',n-1-match,seq,caps)
print(sorted((g['margin'],g['end'][:3]) for g in d if g['result']==-1 and g['opponent']=='strong'))
```