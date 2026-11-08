```python
import json, numpy as np
d=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
rep=[i for i,g in enumerate(d) if g['end']=='repetition']
for gi in rep:
    g=d[gi]; mv=g['moves']
    print(gi,g['opponent'],g['net_color'],g['result'],g['margin'],'last',mv[-1]['by'],[(m['by'][0],m['move']) for m in mv[-9:]])
gi=[i for i in rep if d[i]['margin']>10][0]; n=len(d[gi]['moves'])
rows=[show(B[gi,k]) for k in range(n-9,n)]
for r in range(9): print(' '.join(x[r] for x in rows))
```