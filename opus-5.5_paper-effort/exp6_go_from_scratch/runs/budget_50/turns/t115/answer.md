```python
import sys; sys.path.insert(0,'/work'); from ana import *
print(diag(26))
G,B=load(26)
for i,g in enumerate(G):
    if g['end']!='repetition': continue
    me=1 if g['net_color']=='black' else -1
    m=g['moves'][-1]; p=m['move']; bd=board_before(B,i,m['ply'])
    if m['by']!='net' or p==81: print(m['by'],p); continue
    nb,cap=play(bd,p,me); s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
    print(g['margin'],'cap',cap,'libs',len(ll),'emp',sum(bd[q]==0 for q in NBL[p]),'own',sum(bd[q]==me for q in NBL[p]),dict(m['top5']).get(p))
```