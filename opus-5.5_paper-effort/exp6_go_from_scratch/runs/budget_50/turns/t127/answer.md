```python
import sys; sys.path.insert(0,'/work'); from ana import *
print(diag(30,'strong'))
G,B=load(30); from collections import Counter; c=Counter()
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply'])
        if all(bd[q]==me for q in NBL[p]): c['eyefill']+=1
print(c)
```