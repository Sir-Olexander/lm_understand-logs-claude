```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(26)
fm=Counter(); r=Counter()
for g in G:
    if g['net_color']!='black': continue
    p=g['moves'][0]['move']; fm[divmod(p,9)]+=1; r[(p==40,g['result']>0)]+=1
print(fm.most_common(6), r, G[0]['moves'][0]['top5'] if G[0]['net_color']=='black' else G[1]['moves'][0]['top5'])
# second move of net as black: distance to opp first move
c=Counter()
for g in G:
    if g['net_color']!='black': continue
    o=g['moves'][1]['move']; m=g['moves'][2]['move']
    c[(abs(o//9-m//9)+abs(o%9-m%9))]+=1
print(sorted(c.items()))
```