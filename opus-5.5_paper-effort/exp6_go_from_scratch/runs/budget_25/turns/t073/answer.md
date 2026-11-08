```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
d=json.load(open('/work/pass_10/games.json'))
B=np.load('/work/pass_10/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
for gi in (9,41):
    g=d[gi]; n=len(g['moves']); me=1 if g['net_color']=='black' else 2
    ply=n-3
    prev=tob(B[gi,ply-1]); mv=g['moves'][ply]['move']
    r=G.play(prev,mv,me); g2,l2=G.group(r[0],mv)
    print(gi,ply,divmod(mv,9),'cap',len(r[1]),'libs after',len(l2),'size',len(g2))
    b2=[0 if v==0 else (1 if v==me else 2) for v in prev]
    print(showme(prev,me))
    for q in G.NB[mv]:
        if prev[q]:
            gg,ll=G.group(prev,q); print('  nbr',divmod(q,9),'own' if prev[q]==me else 'opp','size',len(gg),'libs',sorted(divmod(x,9) for x in ll))
```