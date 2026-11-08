```python
import json,numpy as np
G=json.load(open('/work/pass_2/games.json'))
print(G[0].keys())
for g in G:
    if g['opponent']!='strong': continue
    mv=g['moves']; npass=[m['by'] for m in mv if m['move']==81]
    last=mv[-1]
    print(g['net_color'],g['result'],g['margin'],g['end'],len(mv),'passes',[ (m['ply'],m['by'][:1]) for m in mv if m['move']==81][:6],'last by',last['by'])
```