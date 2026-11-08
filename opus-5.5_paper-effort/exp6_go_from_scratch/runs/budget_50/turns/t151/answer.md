```python
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); BASE['c_open']=0.25; json.dump(BASE,open('/work/base.json','w'))
prm=build4.default_prm(); prm.update(BASE); prm.update(c_open=0.35)
n=build4.build(prm,inp,mv); n.save(save); print(n.used, prm['atk'], prm['c_open'])
```