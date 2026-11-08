```python
import json, numpy as np, random
d=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
rng=random.Random(11)
items=[{"board":["........."]*9,"to_move":"black"}]
for _ in range(39):
    gi=rng.randrange(len(d)); n=len(d[gi]['moves'])
    k=rng.randrange(2,n)
    b=B[gi,k-1]
    rows=[''.join({1:'X',-1:'O',0:'.'}[int(b[r*9+c])] for c in range(9)) for r in range(9)]
    items.append({'board':rows,'to_move':'black' if k%2==0 else 'white'})
json.dump(items,open('/work/pos7_items.json','w'))
print(json.dumps({'positions':items,'activations':True},separators=(',',':')))
```