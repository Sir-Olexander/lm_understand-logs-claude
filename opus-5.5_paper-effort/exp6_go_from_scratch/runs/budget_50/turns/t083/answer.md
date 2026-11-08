```python
import sys,json; sys.path.insert(0,'/work'); from ana import *
G,B=load(18)
k=0; pos=[]
for i,g in enumerate(G):
    if not (-6<g['margin']<0): continue
    ps=[m for m in g['moves'] if m['by']=='net' and m['move']==81]
    if not ps: continue
    m=ps[0]; k+=1
    if k>3: break
    pos.append({"moves":[x['move'] if x['move']!=81 else 'pass' for x in g['moves'][:m['ply']]]})
for i,g in enumerate(G):
    if g['margin']>15 and len(pos)<6:
        ps=[m for m in g['moves'] if m['by']=='net' and m['move']==81]
        if ps: pos.append({"moves":[x['move'] if x['move']!=81 else 'pass' for x in g['moves'][:ps[0]['ply']]]})
pos.append({"moves":[40,30,50,31]})
print(json.dumps({"positions":pos,"activations":False}))
```