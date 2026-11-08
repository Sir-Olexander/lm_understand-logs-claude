```python
s=open('/work/build3.py').read().replace("        W[6][u('Terr',p),o]=0  # placeholder (Terr is in L5, needs carry) \n","")
open('/work/build3.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build3
n=build3.build(build3.default_prm(),inp,mv); n.save(save); print(n.used)
```