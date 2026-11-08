```python
import json, numpy as np, random
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
rng=random.Random(1)
items=[]
sel=[]
for _ in range(200):
    gi=rng.randrange(len(d)); n=len(d[gi]['moves'])
    k=rng.randrange(5,n)
    b=B[gi,k-1]
    to_move='black' if k%2==0 else 'white'
    rows=[''.join({1:'X',-1:'O',0:'.'}[int(b[r*9+c])] for c in range(9)) for r in range(9)]
    items.append({'board':rows,'to_move':to_move}); sel.append((gi,k))
json.dump(sel,open('/work/pos2_sel.json','w'))
print(json.dumps({'positions':items,'activations':True})[:300])
open('/work/pos2_req.json','w').write(json.dumps({'positions':items,'activations':True}))
```