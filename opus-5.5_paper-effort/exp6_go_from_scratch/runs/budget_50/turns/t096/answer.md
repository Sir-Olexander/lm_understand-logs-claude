```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(21)
import numpy as np
for col in ('black','white'):
    m=np.array([g['margin'] for g in G if g['net_color']==col])
    print(col, 'losses',np.sort(m[m<0]))
```