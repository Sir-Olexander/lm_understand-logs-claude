```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
ex=[]
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
                if len(g2)>=2 and p!=L and len(ex)<6:
                    b2=(bd*me).astype(int); b2[L]=3
                    ex.append((b2,divmod(L,9),len(g2),divmod(p,9),m['top5'][:3]))
print('\n'.join('  '.join(''.join('.XO*'[x] for x in e[0][r*9:r*9+9]) for e in ex) for r in range(9)))
for e in ex: print(e[1:])
```