```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_10/games.json'))
B=np.load('/work/pass_10/boards.npy')
def showme(b,me):
    ch={0:'.',1:'X' if me==1 else 'O',-1:'O' if me==1 else 'X'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(d):
    if g['end']!='repetition' or g['result']!=-1: continue
    n=len(g['moves']); last=B[gi,n-1]; me=1 if g['net_color']=='black' else 2
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    print(gi,g['opponent'],g['margin'],'len',n-1-match,[(m['by'][0],m['move'],m.get('top5',[[0,0]])[0]) for m in g['moves'][match+1:]],caps)
    if g['margin']>50:
        rows=[showme(B[gi,k],me) for k in range(match,n)]
        for r in range(9): print(' '.join(x[r] for x in rows))
```