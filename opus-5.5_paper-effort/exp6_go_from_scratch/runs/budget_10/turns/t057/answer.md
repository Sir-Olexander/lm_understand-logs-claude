```python
import json, numpy as np, sys, pickle
sys.path.insert(0,'/work')
import go
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
G=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy'); V=np.load('/work/pass_6/valid.npy')
def rows(bd):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9)]
sel=[(37,3),(37,5),(37,11),(37,21),(37,31),(6,15),(6,35),(6,55),(6,75),(13,9),(13,29),(13,49),(13,69),(58,10),(58,30),(58,50),(58,70),(58,90),(64,20),(64,40),(64,60),(64,80),(73,25),(73,45),(73,65),(73,85),(80,14),(80,34),(80,54),(80,74),(88,30),(88,60),(88,90),(17,12),(17,24),(17,40),(26,50),(26,80)]
positions=[{"board":["........."]*9,"to_move":"black"}]; meta=[(-1,-1)]
for gi,k in sel:
    if not V[gi,k]: continue
    positions.append({"board":rows(B[gi,k]),"to_move":'white' if k%2==0 else 'black'}); meta.append((gi,k))
pickle.dump(meta,open('/work/meta7.pkl','wb'))
s=json.dumps({"positions":positions,"activations":True},separators=(',',':'))
print(len(positions),len(s)); print(s)
```