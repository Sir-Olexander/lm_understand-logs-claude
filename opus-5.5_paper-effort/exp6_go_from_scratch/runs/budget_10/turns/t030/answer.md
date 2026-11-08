```python
import json, numpy as np, sys
from collections import Counter
G=json.load(open('/work/pass_2/games.json'))
for opp in ['strong','weak']:
    c0=Counter(); c1=Counter(); lines=Counter()
    for g in G:
        if g['opponent']!=opp: continue
        for m in g['moves']:
            if m['by']=='opponent' and m['move']<81 and m['ply']<20:
                r,c=divmod(m['move'],9); lines[min(r,c,8-r,8-c)]+=1
        if g['net_color']=='white':
            c0[divmod(g['moves'][0]['move'],9)]+=1
    print(opp,'first moves as black',c0.most_common(8))
    print(opp,'line distribution of opp moves ply<20',sorted(lines.items()))
# our line distribution
lines=Counter()
for g in G:
    for m in g['moves']:
        if m['by']=='net' and m['move']<81 and m['ply']<20:
            r,c=divmod(m['move'],9); lines[min(r,c,8-r,8-c)]+=1
print('net',sorted(lines.items()))
```