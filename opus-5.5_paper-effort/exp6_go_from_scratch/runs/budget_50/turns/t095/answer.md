```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(21)
from collections import Counter
c=Counter((g['end'],g['result']) for g in G); print(c)
print(diag(21))
```