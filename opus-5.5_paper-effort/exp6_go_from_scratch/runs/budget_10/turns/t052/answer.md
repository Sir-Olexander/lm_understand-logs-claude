```python
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
P=dict(build.DEF); P['tengen']=1.5
W=build.build(P)
for k,v in W.items(): save(k,v.astype(np.float32))
code=open('/work/build.py').read().replace("tengen=0.4,","tengen=1.5,")
open('/work/build.py','w').write(code)
print('saved')
```