```python
import sys; sys.path.insert(0,'/work'); from ana import *
print(diag(28,'strong'))
G,B=load(28)
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    ps=[m for m in g['moves'] if m['by']=='net' and m['move']==81 and m['ply']<60]
    if ps: print(g['result'],g['margin'],ps[0]['ply'],ps[0]['top5'][:3])
```