```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
gs=[i for i,g in enumerate(G) if g['opponent']=='strong']
i=gs[3]
print(G[i]['net_color'],G[i]['margin'])
print(showmany([B[i,k] for k in (5,11,17,23,29,35,41)]))
print(showmany([B[i,k] for k in (47,53,59,65,71,77,83)]))
```