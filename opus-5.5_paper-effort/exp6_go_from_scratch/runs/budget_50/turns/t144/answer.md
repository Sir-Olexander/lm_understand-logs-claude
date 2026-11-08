```python
import sys; sys.path.insert(0,'/work'); from ana import *
d=diag(36); print({k:d[k] for k in d if k.startswith('rep') or k in ('wins','cap_taken','cap_avail')})
```