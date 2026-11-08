```python
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json'))
prm=build4.default_prm(); prm.update(BASE); prm.update(c_own=0.7)
n=build4.build(prm,inp,mv); n.save(save); print(n.used, prm['c_own'], prm['c_open'])
```