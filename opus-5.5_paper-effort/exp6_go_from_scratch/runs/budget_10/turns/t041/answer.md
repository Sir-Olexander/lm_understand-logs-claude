```python
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
import json
json.dump({k:(v if not isinstance(v,dict) else {str(kk):vv for kk,vv in v.items()}) for k,v in build.DEF.items()},open('/work/params_v3.json','w'))
print('saved')
```