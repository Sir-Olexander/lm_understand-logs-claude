```python
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); print(BASE)
prm=build4.default_prm(); prm.update(BASE)
n=build4.build(prm,inp,mv); n.save(save); print(n.used, prm['c_opp'], prm['c_open'], prm['atk'], prm['iso5'], prm['iso6'], prm['eye'])
```