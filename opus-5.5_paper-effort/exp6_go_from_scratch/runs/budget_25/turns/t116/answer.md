```python
import json, numpy as np, collections
for ps in (17,19):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    rep=[(g['result'],g['margin']) for g in d if g['end']=='repetition']
    print(ps,'rep',rep)
    print(' losses',sorted(g['margin'] for g in d if g['result']==-1))
```