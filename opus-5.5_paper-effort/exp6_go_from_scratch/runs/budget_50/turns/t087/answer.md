```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(18)
DG=[[q for q in range(81) if abs(q//9-p//9)==1 and abs(q%9-p%9)==1] for p in range(81)]
cs={'net':Counter(),'opponent':Counter()}; tot=Counter()
for i,g in enumerate(G):
    me0=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['move']==81 or m['ply']>40: continue
        me=me0 if m['by']=='net' else -me0
        bd=board_before(B,i,m['ply']); p=m['move']
        ao=sum(bd[q]==me for q in NBL[p]); ap=sum(bd[q]==-me for q in NBL[p])
        do=sum(bd[q]==me for q in DG[p]); dp=sum(bd[q]==-me for q in DG[p])
        line=min(p//9,p%9,8-p//9,8-p%9)
        cs[m['by']][('ao',min(ao,3))]+=1; cs[m['by']][('ap',min(ap,3))]+=1; cs[m['by']][('do',min(do,2))]+=1; cs[m['by']][('dp',min(dp,2))]+=1; cs[m['by']][('line',line)]+=1
        tot[m['by']]+=1
for f in sorted(set(cs['net'])|set(cs['opponent'])):
    print(f, round(cs['net'][f]/tot['net'],3), round(cs['opponent'][f]/tot['opponent'],3))
```