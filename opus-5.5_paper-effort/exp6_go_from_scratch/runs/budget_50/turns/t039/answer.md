```python
import sys; sys.path.insert(0,'/work')
import build2
prm=build2.default_prm(); prm.update(scap=0)
n=build2.build(prm,inp,mv); n.save(save); print(n.used)
```