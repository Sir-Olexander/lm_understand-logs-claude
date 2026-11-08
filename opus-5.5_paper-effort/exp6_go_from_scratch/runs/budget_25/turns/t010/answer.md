```python
import json, numpy as np
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy'); V=np.load('/work/pass_1/valid.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9))
for gi in [32,33,45]:
    g=d[gi]; n=len(g['moves'])
    print(gi,g['net_color'],g['margin'],g['end'])
    print(show(B[gi,n-1]))
    for m in g['moves'][-8:]:
        print(m['ply'],m['by'],m['move'],m.get('top5',''))
```