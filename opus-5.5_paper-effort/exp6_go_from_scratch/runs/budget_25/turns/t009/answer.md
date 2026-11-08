```python
import json, numpy as np
d=json.load(open('/work/pass_1/games.json'))
print(type(d), len(d))
g=d[0]; print(g.keys())
print({k:g[k] for k in g if k!='moves'})
print(g['moves'][:6])
B=np.load('/work/pass_1/boards.npy'); V=np.load('/work/pass_1/valid.npy')
print(B.shape,V.shape)
for i,g in enumerate(d):
    if g['opponent']=='strong':
        print(i,g['net_color'],g['result'],g['margin'],g['end'],g['stones_captured'] if 'stones_captured' in g else '', len(g['moves']))
```