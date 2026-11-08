```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(43)
for g in G:
    if g['result']<0: print(g['opponent'],g['net_color'],g['margin'],g['end'],g['moves'][-1]['by'])
```