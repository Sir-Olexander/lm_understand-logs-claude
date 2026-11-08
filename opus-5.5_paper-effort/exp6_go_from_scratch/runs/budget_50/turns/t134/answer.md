```python
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); BASE.update(scap3=0.75,pw2=0.4,kap=0.0,early=[-3,-1.2,0,0,0]); json.dump(BASE,open('/work/base.json','w'))
prm=build4.default_prm(); prm.update(BASE)
n=build4.build(prm,inp,mv); n.save(save); print(n.used, BASE)
```