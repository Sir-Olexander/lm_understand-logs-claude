```python
import json, numpy as np
d=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=0; g=d[gi]
rows=[show(B[gi,k]) for k in range(0,40,4)]
for r in range(9): print(' '.join(x[r] for x in rows))
for m in g['moves'][:40]:
    if m['by']=='net': print(m['ply'],divmod(m['move'],9),[(divmod(a,9),p) for a,p in m['top5'][:4]])
    else: print(m['ply'],'opp',divmod(m['move'],9))
```