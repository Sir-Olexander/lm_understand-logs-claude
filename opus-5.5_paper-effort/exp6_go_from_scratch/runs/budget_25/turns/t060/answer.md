```python
import json, numpy as np
d=json.load(open('/work/pass_8/games.json'))
B=np.load('/work/pass_8/boards.npy')
def showme(b,me):
    ch={0:'.',1:'X' if me==1 else 'O',-1:'O' if me==1 else 'X'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
rep=[i for i,g in enumerate(d) if g['end']=='repetition' and g['result']==-1]
for gi in rep:
    g=d[gi]; mv=g['moves']; me=1 if g['net_color']=='black' else 2
    print(gi,g['opponent'],g['net_color'],g['margin'],len(mv),[(m['by'][0],m['move']) for m in mv[-10:]])
gi=rep[1]; g=d[gi]; n=len(g['moves']); me=1 if g['net_color']=='black' else 2
rows=[showme(B[gi,k],me) for k in range(n-9,n)]
for r in range(9): print(' '.join(x[r] for x in rows))
print([ (m['ply'],m['by'][0],m['move'],m.get('top5',[])[:2]) for m in g['moves'][-9:]])
```