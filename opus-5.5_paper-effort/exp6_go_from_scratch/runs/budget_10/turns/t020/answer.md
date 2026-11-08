```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
losses=[gi for gi,g in enumerate(G) if g['result']==-1]
for gi in losses:
    g=G[gi]; n=len(g['moves'])
    print(gi,g['opponent'],g['net_color'],g['margin'],g['end'],n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
gi=losses[0]
g=G[gi]; n=len(g['moves'])
for k in [20,40,60,n-1]:
    if k<n: print(k); print(show(B[gi,k]))
```