```python
import sys; sys.path.insert(0,'/work')
import build2
prm=build2.default_prm(); prm.update(scap=2)
n=build2.build(prm,inp,mv); n.save(save); print(n.used, n.blk[4])
```