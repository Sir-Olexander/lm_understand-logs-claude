```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(17)
for i,g in enumerate(G):
    if g['end']!='repetition': continue
    me=1 if g['net_color']=='black' else -1
    m=g['moves'][-1]; p=m['move']; bd=board_before(B,i,m['ply'])
    if m['by']!='net': print('opp'); continue
    if p==81: print('pass'); continue
    nb,cap=play(bd,p,me)
    s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
    emp=sum(bd[q]==0 for q in NBL[p])
    adj=[len(l) for c,stn,l in groups(bd) if c==me and any(q in stn for q in NBL[p])]
    print(g['margin'],'cap',cap,'libs_after',len(ll),'size',len(s2),'emp',emp,'ownadj',adj,'prob',dict(m['top5']).get(p),'ply',m['ply'],[x['move'] for x in g['moves'][-6:]])
```