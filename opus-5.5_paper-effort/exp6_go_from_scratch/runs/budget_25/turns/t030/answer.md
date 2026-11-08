```python
import json, numpy as np
d=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
wins=[i for i,g in enumerate(d) if g['opponent']=='strong' and g['margin']>70]
gi=wins[0]; g=d[gi]; n=len(g['moves'])
print(gi,g['net_color'],n,g['stones_captured_by_net'])
rows=[show(B[gi,k]) for k in range(9,min(n,110),10)]
for r in range(9): print(' '.join(x[r] for x in rows))
# passes
print([(m['ply'],m['by']) for m in g['moves'] if m['move']==81])
```