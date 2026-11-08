```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(18); c=Counter()
def regions(bd):
    seen=set(); out=[]
    for p in range(81):
        if bd[p]!=0 or p in seen: continue
        st=[p]; seen.add(p); border=set(); k=0
        while k<len(st):
            q=st[k]; k+=1
            for m in NBL[q]:
                if bd[m]==0 and m not in seen: seen.add(m); st.append(m)
                elif bd[m]!=0: border.add(int(bd[m]))
        out.append((st,border))
    return out
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    close= -8<g['margin']<0
    for m in g['moves']:
        if m['by']=='net' and m['move']==81:
            bd=board_before(B,i,m['ply'])
            neutral=sum(len(st) for st,bo in regions(bd) if len(bo)==2)
            c[(close,'pass_with_neutral>0')]+= neutral>0; c[(close,'passes')]+=1; c[(close,'neutral_pts')]+=neutral
print(c)
```