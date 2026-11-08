```python
import sys; sys.path.insert(0,'/work'); from ana import *
print(diag(31))
G,B=load(31)
for i,g in enumerate(G):
    if g['end']!='repetition': continue
    me=1 if g['net_color']=='black' else -1
    m=g['moves'][-1]; p=m['move']; bd=board_before(B,i,m['ply'])
    if m['by']!='net': print('opp',g['result']); continue
    if p==81: print('pass?'); continue
    nb,cap=play(bd,p,me); s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
    print(g['margin'],m['ply'],'cap',cap,'libs',len(ll),'emp',sum(bd[q]==0 for q in NBL[p]),'own',sum(bd[q]==me for q in NBL[p]),m['top5'][:3])
```