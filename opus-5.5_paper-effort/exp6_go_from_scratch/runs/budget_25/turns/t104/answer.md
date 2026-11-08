```python
import json, numpy as np, collections
for ps in (13,14,16):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    gs=[g for g in d if g['opponent']=='strong']
    for col in ('black','white'):
        L=sorted(g['margin'] for g in gs if g['net_color']==col and g['result']==-1)
        W=[g['margin'] for g in gs if g['net_color']==col and g['result']==1]
        print(ps,col,'wins',len(W),'wipe',sum(1 for x in W if x>60),'losses',L)
```