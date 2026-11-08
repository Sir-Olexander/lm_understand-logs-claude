```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def analyze(k):
    G=json.load(open(f'/work/pass_{k}/games.json'))
    B=np.load(f'/work/pass_{k}/boards.npy')
    stats=Counter()
    for gi,g in enumerate(G):
        if g['opponent']!='strong': continue
        me=1 if g['net_color']=='black' else 2; op=3-me
        for kk,m in enumerate(g['moves']):
            if m['by']!='net': continue
            b=conv(B[gi,kk-1]) if kk>0 else np.zeros(81,np.int8)
            # available captures
            best=0; seen=set()
            for i in range(81):
                if b[i]==op and i not in seen:
                    Gs,L=go.group(b,i); seen|=Gs
                    if len(L)==1: best=max(best,len(Gs))
            if best==0: continue
            after=conv(B[gi,kk])
            captured=int(((b==op)&(after==0)).sum())
            bucket='1' if best==1 else ('2-3' if best<=3 else '4+')
            stats[(bucket,'avail')]+=1
            if captured>0: stats[(bucket,'took')]+=1
            if captured>=best: stats[(bucket,'tookmax')]+=1
    return stats
for k in (2,4,5,6,8):
    s=analyze(k)
    print(k,{b:(s[(b,'avail')], round(s[(b,'took')]/max(1,s[(b,'avail')]),2), round(s[(b,'tookmax')]/max(1,s[(b,'avail')]),2)) for b in ('1','2-3','4+')})
```