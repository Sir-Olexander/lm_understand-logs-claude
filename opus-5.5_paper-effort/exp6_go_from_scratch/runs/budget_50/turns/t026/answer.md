```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(5)
ex=[]
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    for m in g['moves']:
        if m['by']=='net' and m['move']==81 and m['ply']<60:
            ex.append((i,m['ply'],m['top5'][:4])); break
print(len(ex))
sel=ex[:5]
print(showmany([board_before(B,i,p) for i,p,_ in sel]))
for e in sel: print(G[e[0]]['net_color'],e[1:])
```