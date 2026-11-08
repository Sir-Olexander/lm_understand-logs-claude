```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
ex={'own_libs2_e0':[],'noown_e1':[]}
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
        if key in ex and len(ex[key])<4:
            b2=bd.copy()*me; b2=b2.astype(int); b2[p]=3  # own=X
            ex[key].append((b2,m['top5'][:3],divmod(p,9)))
for k,v in ex.items():
    print(k)
    print('\n'.join('  '.join(''.join('.XO*'[x] for x in b[r*9:r*9+9]) for b,_,_ in v) for r in range(9)))
    for _,t,p in v: print(p,t)
```