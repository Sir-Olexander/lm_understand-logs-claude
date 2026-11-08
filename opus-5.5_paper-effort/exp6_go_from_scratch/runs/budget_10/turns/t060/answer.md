```python
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
code=open('/work/build.py').read().replace("w_cc=3.0,","w_cc=4.0,")
open('/work/build.py','w').write(code)
importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
print(build.DEF['w_cc'])
```