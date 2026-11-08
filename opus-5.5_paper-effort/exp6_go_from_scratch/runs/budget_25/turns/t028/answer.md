```python
import json, numpy as np
d=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
losses=[i for i,g in enumerate(d) if g['opponent']=='strong' and g['result']==-1]
for gi in losses[:6]:
    g=d[gi]; n=len(g['moves'])
    rows=[show(B[gi,k]) for k in (19,39,59,n-1)]
    print(gi,g['net_color'],g['margin'],g['end'],n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
    for r in range(9): print('   '.join(x[r] for x in rows))
```