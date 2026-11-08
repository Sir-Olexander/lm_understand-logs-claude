```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(11); cat=Counter()
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply'])
        nb,cap=play(bd,p,me)
        s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
        if not(cap==0 and len(ll)==1): continue
        pr=dict(m['top5']).get(p,0)
        gr=groups(bd); adj=[(stn,l) for c,stn,l in gr if c==me and any(q in stn for q in NBL[p])]
        nst=[sum(q in stn for q in NBL[p]) for stn,l in adj]
        libs=[len(l) for stn,l in adj]
        emp=sum(bd[q]==0 for q in NBL[p])
        oppat=any(len(l)==1 for c,stn,l in gr if c==-me and any(q in stn for q in NBL[p]))
        key=('multi' if any(k>=2 for k in nst) else 'single')+'_e%d_libs%s'%(emp,sorted(libs))
        cat[(key, pr>0.5)]+=len(s2)
print(sum(cat.values()))
for k,v in cat.most_common(25): print(k,v)
```