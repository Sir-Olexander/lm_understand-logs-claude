```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(5)
DG=[[q for q in range(81) if abs(q//9-p//9)==1 and abs(q%9-p%9)==1] for p in range(81)]
def feat(bd,p,me):
    a_own=sum(bd[q]==me for q in NBL[p]); a_opp=sum(bd[q]==-me for q in NBL[p])
    d_own=sum(bd[q]==me for q in DG[p]); d_opp=sum(bd[q]==-me for q in DG[p])
    return (min(a_own,2),min(a_opp,2),min(d_own,2),min(d_opp,2))
cs={'net':Counter(),'opponent':Counter()}
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me0=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['move']==81 or m['ply']>50 or m['ply']<6: continue
        me=me0 if m['by']=='net' else -me0
        cs[m['by']][feat(board_before(B,i,m['ply']),m['move'],me)]+=1
for k in cs:
    t=sum(cs[k].values()); print(k,t)
    print([(f,round(v/t,3)) for f,v in cs[k].most_common(14)])
```