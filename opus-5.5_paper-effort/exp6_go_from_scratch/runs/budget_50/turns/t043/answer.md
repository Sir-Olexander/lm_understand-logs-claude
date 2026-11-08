```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    found=False
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        bd=board_before(B,i,m['ply'])
        for c,stn,l in groups(bd):
            if c==me and len(l)==1 and len(stn)>=2:
                print(show(bd*me)); print(sorted(divmod(x,9) for x in stn), [divmod(x,9) for x in l]); found=True; break
        if found: break
    if found: break
```