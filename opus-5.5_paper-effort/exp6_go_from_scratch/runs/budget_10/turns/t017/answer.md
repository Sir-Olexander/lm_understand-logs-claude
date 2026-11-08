```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
for gi,g in enumerate(G):
    if g['end']=='repetition' and g['result']==-1:
        n=len(g['moves'])
        print('game',gi,g['opponent'],g['net_color'],g['result'],g['margin'])
        for m in g['moves'][-9:]:
            mv=m['move']; print(m['ply'],m['by'],divmod(mv,9) if mv<81 else 'pass', [ (divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])][:3])
        for k in range(n-9,n):
            print(k); print(show(B[gi,k]))
```