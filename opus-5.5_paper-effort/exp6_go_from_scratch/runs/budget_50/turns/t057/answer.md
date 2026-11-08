```python
s=open('/work/build2.py').read()
s=s.replace("W[5][u('S',q),j]+=2-2; W[5][u('O',q),j]+=2","W[5][u('S',q),j]-=1; W[5][u('O',q),j]+=2")
open('/work/build2.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build2
n=build2.build(build2.default_prm(),inp,mv); n.save(save); print(n.used)
```