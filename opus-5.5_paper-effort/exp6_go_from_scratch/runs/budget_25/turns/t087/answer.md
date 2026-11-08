```python
import json, numpy as np, collections
d=json.load(open('/work/pass_13/games.json'))
for opp in ('strong','weak'):
    gs=[g for g in d if g['opponent']==opp]
    print(opp,collections.Counter((g['end'][:3],g['result']) for g in gs), 'wipe',sum(1 for g in gs if g['margin']>60))
    print('  losses',sorted((g['margin'],g['end'][:3],g['net_color'][0]) for g in gs if g['result']==-1))
B=np.load('/work/pass_13/boards.npy')
for gi,g in enumerate(d):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    print(gi,g['result'],g['margin'],'len',n-1-match,[(m['by'][0],m['move'],round(dict(m.get('top5',[])).get(m['move'],0),2)) for m in g['moves'][match+1:]],caps)
```