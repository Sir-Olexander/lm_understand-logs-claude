```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
for gi in [18,49]:
    g=G[gi]
    print('game',gi,g['net_color'])
    for m in g['moves'][:24]:
        mv=m['move']
        print(m['ply'],m['by'][:3],divmod(mv,9) if mv<81 else 'P', [(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])][:4])
    print(show(B[gi,23]))
```