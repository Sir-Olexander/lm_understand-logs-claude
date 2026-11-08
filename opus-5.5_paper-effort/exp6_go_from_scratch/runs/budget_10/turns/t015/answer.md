```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
print(len(G)); print(G[0].keys())
print({k:G[0][k] for k in G[0] if k!='moves'})
print(G[0]['moves'][:3])
for g in G:
    if g['end']!='two_passes' and g['end']!='two passes':
        print(g['opponent'],g['net_color'],g['result'],g['margin'],g['end'],len(g['moves']))
```