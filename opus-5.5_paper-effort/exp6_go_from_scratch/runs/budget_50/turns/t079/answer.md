```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(18)
import numpy as np
m=np.array([g['margin'] for g in G]); c=np.array([g['net_color']=='black' for g in G])
print(np.sort(m[m<0])); print('black losses',np.sort(m[(m<0)&c]))
print(diag(18))
```