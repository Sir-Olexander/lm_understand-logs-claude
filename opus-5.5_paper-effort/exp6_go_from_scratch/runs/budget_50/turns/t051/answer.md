```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(11)
ex=[]
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        p=m['move']; bd=board_before(B,i,m['ply'])
        nb,cap=play(bd,p,me)
        s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
        if cap==0 and len(ll)==1 and len(s2)>=3 and len(ex)<6:
            b2=(bd*me).astype(int); b2[p]=3; ex.append((b2,divmod(p,9),len(s2),m['top5'][:3],i,m['ply']))
print('\n'.join('  '.join(''.join('.XO*'[x] if x!=-1 else 'O' for x in e[0][r*9:r*9+9]) for e in ex) for r in range(9)))
for e in ex: print(e[1:])
```