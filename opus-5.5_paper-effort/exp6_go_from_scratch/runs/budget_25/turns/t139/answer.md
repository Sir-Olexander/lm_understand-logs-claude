```python
import json
for ps in (17,19,20,21):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    gs=[g for g in d if g['opponent']=='strong']
    rl=[g['margin'] for g in gs if g['end']=='repetition' and g['result']==-1 and g['margin']>0]
    print(ps,'score',sum(g['result']==1 for g in gs)/len(gs),'wipe',sum(1 for g in gs if g['margin']>60)/len(gs),'rep losses winning',len(rl))
```