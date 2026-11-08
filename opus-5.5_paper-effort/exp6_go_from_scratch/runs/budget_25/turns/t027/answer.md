```python
import json, numpy as np, collections
d=json.load(open('/work/pass_3/games.json'))
for opp in ('strong','weak'):
    gs=[g for g in d if g['opponent']==opp]
    c=collections.Counter((g['end'],g['result']) for g in gs)
    print(opp,c)
    print(sorted((g['margin'],g['result'],g['end'][:3],g['net_color'][0]) for g in gs))
```