```python
import json, numpy as np
d=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
losses=[i for i,g in enumerate(d) if g['opponent']=='strong' and g['result']==-1]
for gi in losses[6:12]:
    g=d[gi]; n=len(g['moves'])
    rows=[show(B[gi,k]) for k in (11,23,35,47,n-1)]
    print(gi,'net',g['net_color'],g['margin'],n)
    for r in range(9): print('   '.join(x[r] for x in rows))
```