```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
gs=[i for i,g in enumerate(G) if g['opponent']=='strong']
sel=gs[1:5]+gs[33:35]
print(showmany([B[i,len(G[i]['moves'])-1] for i in sel]))
print([ (G[i]['net_color'],G[i]['margin']) for i in sel])
g=G[gs[1]]
print([(m['move'],m['by'][0]) for m in g['moves'][:30]])
```