```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(11)
L=[i for i,g in enumerate(G) if g['result']<0 and g['net_color']=='black']
i=L[3]; print(G[i]['margin'])
print(showmany([B[i,k] for k in (3,7,11,15,19,23,27)]))
print(showmany([B[i,k] for k in (31,35,39,43,47,51,55)]))
```