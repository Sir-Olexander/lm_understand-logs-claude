```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_20/games.json')); B=np.load('/work/pass_20/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
shown=0
for gi,g in enumerate(d):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]; me=1 if g['net_color']=='black' else 2
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    seq=[(m['by'][0],divmod(m['move'],9),round(dict(m.get('top5',[])).get(m['move'],0),2)) for m in g['moves'][match+1:]]
    print(gi,g['result'],g['margin'],'len',n-1-match,seq,caps)
    if g['result']==-1 and shown<3 and g['margin']>0:
        ply=match+1 if g['moves'][match+1]['by']=='net' else match+2
        print(showme(tob(B[gi,ply-1]),me)); shown+=1
```