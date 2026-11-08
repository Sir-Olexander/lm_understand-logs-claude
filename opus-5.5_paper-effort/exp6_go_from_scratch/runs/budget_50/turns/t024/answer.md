```python
import sys; sys.path.insert(0,'/work')
import build
prm=build.default_prm(); prm.update(cap=12,esc=10,uns=12)
n=build.build(prm,inp,mv); n.save(save); print('ok')
```