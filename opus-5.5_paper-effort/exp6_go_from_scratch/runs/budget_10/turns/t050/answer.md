```python
import json, numpy as np, sys
from collections import Counter
G=json.load(open('/work/pass_5/games.json'))
c0=Counter(); lines=Counter(); 
for g in G:
    if g['net_color']=='black': c0[divmod(g['moves'][0]['move'],9)]+=1
    for m in g['moves']:
        if m['by']=='net' and m['move']<81 and m['ply']<20:
            r,c=divmod(m['move'],9); lines[min(r,c,8-r,8-c)]+=1
print(c0.most_common(10)); print(sorted(lines.items()))
print(G[0]['moves'][0]['top5'])
# margin distribution
ms=[g['margin'] for g in G if g['opponent']=='strong']
print(sorted(ms))
```