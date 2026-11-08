```python
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4,U3=8,**{'def':6})
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```