```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
for gi in [27,78]:
    g=G[gi]; n=len(g['moves'])
    first=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
    print(gi,g['net_color'],g['margin'],'pass plies',first)
    k=first[0]
    print(show(B[gi,k-1])); print(go.score(conv(B[gi,k-1])))
    m=g['moves'][k]; print([(divmod(a,9) if a<81 else 'P',p) for a,p in m['top5']])
    print(show(B[gi,n-1])); print(go.score(conv(B[gi,n-1])))
```