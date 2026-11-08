```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
for gi,ply in [(7,98),(41,72),(48,63),(1,59),(3,81)]:
    g=d[gi]; me=1 if g['net_color']=='black' else 2
    m=g['moves'][ply]
    prev=tob(B[gi,ply-1])
    r=G.play(prev,m['move'],me)
    g2,l2=G.group(r[0],m['move']) if r else (None,None)
    print(gi,ply,m['by'],'move',divmod(m['move'],9),'top',[(divmod(a,9),p) for a,p in m['top5'][:3]],'captured',len(r[1]) if r else None,'libs after',len(l2) if r else None,'size',len(g2) if r else None)
    print(showme(prev,me))
```