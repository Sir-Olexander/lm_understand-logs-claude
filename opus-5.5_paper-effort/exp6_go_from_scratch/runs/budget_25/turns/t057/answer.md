```python
import json, numpy as np, collections
d=json.load(open('/work/pass_8/games.json'))
for opp in ('strong','weak'):
    gs=[g for g in d if g['opponent']==opp]
    print(opp,collections.Counter((g['end'],g['result']) for g in gs))
    print(sorted((g['margin'],g['end'][:3]) for g in gs if g['result']==-1))
    print('wipeouts',sum(1 for g in gs if g['margin']>60))
# who passes: count passes by net
gs=[g for g in d if g['opponent']=='strong']
np_=[sum(1 for m in g['moves'] if m['by']=='net' and m['move']==81) for g in gs]
op_=[sum(1 for m in g['moves'] if m['by']=='opponent' and m['move']==81) for g in gs]
print(np.mean(np_),np.mean(op_))
```