```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(11); c=Counter(); ga=Counter()
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        bd=board_before(B,i,m['ply']); nb,cap=play(bd,m['move'],me)
        if cap: c[min(cap,10)]+=cap; ga[(g['result']>0,min(cap,10))]+=1
print(sorted(c.items())); print(sorted(ga.items()))
```