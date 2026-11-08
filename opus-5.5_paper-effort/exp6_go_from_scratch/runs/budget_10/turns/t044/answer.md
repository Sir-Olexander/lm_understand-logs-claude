```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_5/games.json'))
B=np.load('/work/pass_5/boards.npy')
def show(bd,mark=None):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(('*' if r*9+c==mark else ch[int(bd[r*9+c])]) for c in range(9)) for r in range(9))
for gi in [5,11,20,30]:
    g=G[gi]; n=len(g['moves'])
    # find earlier identical board
    last=B[gi,n-1]
    same=[k for k in range(n-1) if np.array_equal(B[gi,k],last)]
    print('game',gi,g['net_color'],'n',n,'same as plies',same)
    for m in g['moves'][same[-1]-1:]:
        mv=m['move']; print(m['ply'],m['by'][:3],divmod(mv,9) if mv<81 else 'P',[(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])][:3])
    for k in range(same[-1]-1,n):
        print(k); print(show(B[gi,k], g['moves'][k]['move']))
```