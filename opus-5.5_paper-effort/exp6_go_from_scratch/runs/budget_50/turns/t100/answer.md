```python
import sys; sys.path.insert(0,'/work'); from ana import *
import numpy as np
G,B=load(21)
for res in (1,-1):
    gs=[g for g in G if g['result']==res]
    print(res,len(gs),'lost',np.mean([g['stones_lost_by_net'] for g in gs]),'capt',np.mean([g['stones_captured_by_net'] for g in gs]))
gs=[g for g in G if g['result']<0]
print(sorted([(g['stones_lost_by_net'],g['stones_captured_by_net'],g['margin']) for g in gs]))
```