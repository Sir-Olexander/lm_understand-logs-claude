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
    st=Counter()
    for gi,g in enumerate(G):
        if g['opponent']!='strong': continue
        me=1 if g['net_color']=='black' else 2; op=3-me
        for kk,m in enumerate(g['moves']):
            if m['by']!='net' or kk==0: continue
            b=conv(B[gi,kk-1])
            seen=set(); groups=[]
            for i in range(81):
                if b[i]==me and i not in seen:
                    Gs,L=go.group(b,i); seen|=Gs
                    if len(L)==1: groups.append((Gs,list(L)[0]))
            if not groups: continue
            # can any be saved by playing its liberty (resulting libs>=2)?
            best=0
            for Gs,lib in groups:
                gg=go.Game(); gg.b=b.copy(); gg.turn=me
                if not gg.legal(lib): continue
                gg.play(lib); G2,L2=go.group(gg.b,lib)
                if len(L2)>=2: best=max(best,len(Gs))
            if best==0: continue
            after=conv(B[gi,kk])
            bucket='1' if best==1 else ('2-3' if best<=3 else '4+')
            st[(bucket,'avail')]+=1
            # saved? check biggest such group still present and has >=2 libs after our move
            ok=False
            for Gs,lib in groups:
                if len(Gs)==best:
                    i=next(iter(Gs))
                    if after[i]==me and len(go.group(after,i)[1])>=2: ok=True
            if ok: st[(bucket,'saved')]+=1
    return st
for k in (4,5,6,8):
    s=analyze(k)
    print(k,{b:(s[(b,'avail')], round(s[(b,'saved')]/max(1,s[(b,'avail')]),2)) for b in ('1','2-3','4+')})
```