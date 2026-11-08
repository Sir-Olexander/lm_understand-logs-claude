```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
items=[];meta=[]
for gi,g in enumerate(d):
    if g['end']!='repetition' or g['result']!=-1: continue
    n=len(g['moves']); last=B[gi,n-1]
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    for ply in range(match+1,n):
        if g['moves'][ply]['by']=='net':
            b=B[gi,ply-1]
            rows=[''.join({1:'X',-1:'O',0:'.'}[int(b[r*9+c])] for c in range(9)) for r in range(9)]
            items.append({'board':rows,'to_move':g['net_color']}); meta.append((gi,ply,g['moves'][ply]['move']))
print(len(items))
json.dump({'items':items,'meta':meta},open('/work/pos10.json','w'))
print(json.dumps({'positions':items,'activations':True},separators=(',',':')))
```