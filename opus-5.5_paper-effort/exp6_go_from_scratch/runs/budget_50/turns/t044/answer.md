```python
import sys, json; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
out=[]
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply'])
        for c,stn,l in groups(bd):
            if c==me and len(l)==1 and len(stn)>=2:
                L=list(l)[0]; nb,cap=play(bd,L,me)
                g2=[ll for cc,s2,ll in groups(nb) if L in s2][0]
                if len(g2)>=2 and p!=L and len(out)<6:
                    out.append((i,m['ply'],L,sorted(stn)))
for i,ply,L,stn in out[:4]:
    print(i,ply,L,stn)
    print(json.dumps([m['move'] if m['move']!=81 else 'pass' for m in G[i]['moves'][:ply]]))
```