```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(21)
sel=[i for i,g in enumerate(G) if g['margin']==-0.5][:6]
print([(G[i]['net_color'],G[i]['end'],[ (m['by'][0]) for m in G[i]['moves'][-3:]]) for i in sel])
print(showmany([B[i,len(G[i]['moves'])-1] for i in sel]))
```