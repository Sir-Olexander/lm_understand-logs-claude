```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(27)
for i,g in enumerate(G):
    if g['end']!='repetition' or g['result']>0: continue
    me=1 if g['net_color']=='black' else -1
    m=g['moves'][-1]; p=m['move']; bd=board_before(B,i,m['ply'])
    nb,cap=play(bd,p,me); s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
    print(g['opponent'],g['margin'],'ply',m['ply'],'cap',cap,'libs',len(ll),'emp',sum(bd[q]==0 for q in NBL[p]),'own',sum(bd[q]==me for q in NBL[p]),m['top5'][:3],[x['move'] for x in g['moves'][-7:]])
    n=len(g['moves']); print(showmany([B[i,k]*me for k in range(n-7,n)]))
```