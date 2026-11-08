```python
import json, numpy as np, sys, pickle
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
def rows(bd):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9)]
sel=[]
games=[18,20,23,49,52,58,63,64]
plies=[0,1,2,4,7,10,15,22,30,45,60,75]
for gi in games:
    for k in plies:
        if V[gi,k]: sel.append((gi,k))
sel=sel[:90]
positions=[]
meta=[]
for gi,k in sel:
    bd=B[gi,k]
    to_move='white' if k%2==0 else 'black'   # board after ply k; ply 0 is black's move
    positions.append({"board":rows(bd),"to_move":to_move}); meta.append((gi,k))
positions.insert(0,{"board":["........."]*9,"to_move":"black"}); meta.insert(0,(-1,-1))
pickle.dump(meta,open('/work/meta3.pkl','wb'))
s=json.dumps({"positions":positions,"activations":True},separators=(',',':'))
print(len(positions),len(s))
print(s)
```