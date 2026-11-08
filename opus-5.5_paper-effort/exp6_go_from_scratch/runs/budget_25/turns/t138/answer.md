```python
import json, numpy as np, collections
for ps in (22,23,24):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    gs=[g for g in d if g['opponent']=='strong']
    rl=[g['margin'] for g in gs if g['end']=='repetition' and g['result']==-1]
    rw=[g['margin'] for g in gs if g['end']=='repetition' and g['result']==1]
    tl=[g['margin'] for g in gs if g['end']!='repetition' and g['result']==-1]
    print(ps,len(gs),'rep losses',sorted(rl),'rep wins',len(rw),'other losses',len(tl),'wipe',sum(1 for g in gs if g['margin']>60))
```