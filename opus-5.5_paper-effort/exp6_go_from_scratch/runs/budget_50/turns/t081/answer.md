```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(18)
k=0
for i,g in enumerate(G):
    if not (-6<g['margin']<0): continue
    me=1 if g['net_color']=='black' else -1
    ps=[m for m in g['moves'] if m['by']=='net' and m['move']==81]
    if not ps: continue
    m=ps[0]; bd=board_before(B,i,m['ply'])
    k+=1
    if k>3: break
    print(g['net_color'],g['margin'],'ply',m['ply'],'top5',m['top5'])
    print(showmany([bd, B[i,len(g['moves'])-1]]))
```