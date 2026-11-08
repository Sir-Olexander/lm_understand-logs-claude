```python
import json, numpy as np
d=json.load(open('/work/pass_16/games.json'))
B=np.load('/work/pass_16/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
cand=[i for i,g in enumerate(d) if g['opponent']=='strong' and g['net_color']=='black' and -6<g['margin']<0]
for gi in cand[:2]:
    g=d[gi]; n=len(g['moves'])
    rows=[show(B[gi,k]) for k in list(range(7,n,12))+[n-1]]
    print(gi,g['margin'],n)
    for r in range(9): print(' '.join(x[r] for x in rows))
    print([(m['ply'],m['by'][0],divmod(m['move'],9),round(dict(m.get('top5',[])).get(m['move'],-1),2)) for m in g['moves'][:24]])
```