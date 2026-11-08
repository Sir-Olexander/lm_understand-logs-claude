```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def show(b,mark=None):
    return '\n'.join(''.join('*' if r*9+c==mark else '.XO'[b[r*9+c]] for c in range(9)) for r in range(9))
for gi in [59,17,34]:
    g=G[gi]; n=len(g['moves'])
    me=1 if g['net_color']=='black' else 2
    passes=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
    print('game',gi,g['net_color'],g['margin'],'net passes at',passes)
    k=passes[0]
    b=conv(B[gi,k-1]); print(show(b)); print('score at first pass',go.score(b))
    print([(divmod(a,9) if a<81 else 'P',round(p,3)) for a,p in g['moves'][k]['top5']])
    # dame: empty points adjacent to both colors
    dame=[divmod(i,9) for i in range(81) if b[i]==0 and any(b[j]==1 for j in go.NB[i]) and any(b[j]==2 for j in go.NB[i])]
    print('dame',dame)
    print(show(conv(B[gi,n-1]))); print(go.score(conv(B[gi,n-1])))
```