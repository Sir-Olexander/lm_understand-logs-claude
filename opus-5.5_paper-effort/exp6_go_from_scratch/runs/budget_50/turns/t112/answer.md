```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(25); c=Counter(); ex=[]
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net': continue
        p=m['move']; bd=board_before(B,i,m['ply'])
        caps={}
        for col,stn,l in groups(bd):
            if col==-me and len(l)==1:
                L=list(l)[0]; caps[L]=caps.get(L,0)+len(stn)
        if not caps: continue
        best=max(caps.values())
        if p in caps: c['taken']+=1; continue
        nb0=play(bd,p,me)[1] if p!=81 else 0
        # properties of best missed capture point
        L=max(caps,key=caps.get); nb,cap=play(bd,L,me)
        libs=len([ll for cc,s2,ll in groups(nb) if L in s2][0])
        own=sum(bd[q]==me for q in NBL[L]); emp=sum(bd[q]==0 for q in NBL[L])
        key=('size%d'%min(best,4),'libs_after%d'%min(libs,3),'own%d'%min(own,2),'emp%d'%emp,'pass' if p==81 else ('othercap' if nb0 else 'other'))
        c[key]+=1
        if best>=3 and len(ex)<4: ex.append((i,m['ply'],L,p,m['top5'][:3],best))
print(c['taken'])
for k,v in c.most_common(22):
    if k!='taken': print(k,v)
print(ex)
```