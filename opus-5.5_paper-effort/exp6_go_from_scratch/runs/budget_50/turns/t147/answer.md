```python
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); print(BASE)
prm=build4.default_prm(); prm.update(BASE); prm.update(atk=8)
n=build4.build(prm,inp,mv); n.save(save); print(n.used, prm['iso6'], prm['iso5'], prm['eye'], prm['atk'])
```