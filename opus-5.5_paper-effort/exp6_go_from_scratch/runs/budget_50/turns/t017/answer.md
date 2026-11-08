```python
import sys; sys.path.insert(0,'/work'); from ana import *
import json
G,B=load(2)
gs=[i for i,g in enumerate(G) if g['opponent']=='strong']
g=G[gs[3]]
print(json.dumps([m['move'] if m['move']!=81 else 'pass' for m in g['moves'][:12]]))
print(json.dumps([m['move'] if m['move']!=81 else 'pass' for m in g['moves'][:24]]))
```