```python
import json, numpy as np
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9))
rep=[i for i,g in enumerate(d) if g['end']=='repetition']
for gi in rep:
    g=d[gi]; mv=g['moves']
    last=mv[-1]
    print(gi,g['opponent'],g['net_color'],g['result'],'last by',last['by'],'seq',[ (m['by'][0],m['move']) for m in mv[-10:]])
gi=rep[0]; n=len(d[gi]['moves'])
for k in range(n-6,n):
    print(k, d[gi]['moves'][k]['by'], d[gi]['moves'][k]['move']); print(show(B[gi,k]))
```