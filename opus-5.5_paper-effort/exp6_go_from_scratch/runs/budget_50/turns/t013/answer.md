```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
from collections import Counter
cat=Counter(); probs=[]
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply']); gr=groups(bd)
        nb,cap=play(bd,p,me)
        g2=[l for c,st,l in groups(nb) if p in st][0]
        if len(g2)!=1 or cap: continue
        own=[ (len(st),len(l)) for c,st,l in gr if c==me and any(q in st for q in NBL[p])]
        emp=sum(bd[q]==0 for q in NBL[p])
        key=('noown' if not own else 'own_libs'+''.join(sorted(str(min(l,3)) for s,l in own)))+'_e%d'%emp
        cat[key]+=1; probs.append(dict(m['top5']).get(p,0))
print(cat.most_common(20)); print(np.histogram(probs,[0,.05,.1,.2,.4,.6,1.01]))
```