```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, random
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
rng=random.Random(7)
items=[]; sel=[]
def tob(b): return [1 if x==1 else (2 if x==-1 else 0) for x in b]
tries=0
while len(items)<60:
    gi=rng.randrange(len(d)); n=len(d[gi]['moves'])
    k=rng.randrange(8,n)
    b=B[gi,k-1]; bb=tob(b)
    gid,groups=G.all_groups(bb)
    nat=sum(1 for g,l,c in groups if len(l)==1 and len(g)>=2)
    if nat==0 and rng.random()<0.6: continue
    to_move='black' if k%2==0 else 'white'
    rows=[''.join({1:'X',-1:'O',0:'.'}[int(b[r*9+c])] for c in range(9)) for r in range(9)]
    items.append({'board':rows,'to_move':to_move}); sel.append((gi,k))
json.dump(items,open('/work/pos2_items.json','w'))
print(json.dumps({'positions':items,'activations':True},separators=(',',':')))
```