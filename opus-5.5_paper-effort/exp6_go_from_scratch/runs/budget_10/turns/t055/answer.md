```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def show(b,mark=None):
    return '\n'.join(''.join('*' if r*9+c==mark else '.XO'[b[r*9+c]] for c in range(9)) for r in range(9))
gi=37; g=G[gi]
for m in g['moves'][:40]:
    mv=m['move']; print(m['ply'],m['by'][:3],divmod(mv,9) if mv<81 else 'P',[(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])][:4])
for k in [9,19,29,39]:
    print(k); print(show(conv(B[gi,k])))
```