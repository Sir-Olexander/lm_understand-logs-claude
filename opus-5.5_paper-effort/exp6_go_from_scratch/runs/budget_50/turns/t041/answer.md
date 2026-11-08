```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(7)
st=Counter()
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply']); gr=groups(bd)
        # savable own atari groups (size>=2)
        for c,stn,l in gr:
            if c==me and len(l)==1 and len(stn)>=2:
                L=list(l)[0]; nb,cap=play(bd,L,me)
                g2=[ll for cc,s2,ll in groups(nb) if L in s2][0]
                if len(g2)>=2:
                    st['savable']+=1; st['saved']+= (p==L)
        nb,cap=play(bd,p,me)
        g2=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
        if cap==0 and len(g2[1])==1: st['selfatari']+=1; st['selfatari_sz']+=len(g2[0])
        if cap==0 and len(g2[1])==2: st['two_lib_result']+=1
        st['moves']+=1
print(st)
```