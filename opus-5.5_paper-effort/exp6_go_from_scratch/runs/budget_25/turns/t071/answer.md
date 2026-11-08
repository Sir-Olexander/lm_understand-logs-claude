```python
import json, numpy as np, collections
d=json.load(open('/work/pass_10/games.json'))
for opp in ('strong','weak'):
    gs=[g for g in d if g['opponent']==opp]
    print(opp,collections.Counter((g['end'][:3],g['result']) for g in gs), 'wipe',sum(1 for g in gs if g['margin']>60))
    print('  losses',sorted((g['margin'],g['end'][:3],g['net_color'][0]) for g in gs if g['result']==-1))
```