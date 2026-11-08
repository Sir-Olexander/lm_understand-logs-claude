```python
import json, numpy as np
d=json.load(open('/work/pass_8/games.json'))
tops=[];chosen_rank=[]
for g in d:
    if g['opponent']!='strong': continue
    for m in g['moves']:
        if m['by']=='net' and m['top5']:
            tops.append(m['top5'][0][1])
            rk=[a for a,p in m['top5']]
            chosen_rank.append(rk.index(m['move']) if m['move'] in rk else 5)
tops=np.array(tops)
print('mean top1',tops.mean(), 'frac top1<0.5',(tops<0.5).mean(),'frac<0.3',(tops<0.3).mean())
print('chosen rank dist',np.bincount(chosen_rank)/len(chosen_rank))
```