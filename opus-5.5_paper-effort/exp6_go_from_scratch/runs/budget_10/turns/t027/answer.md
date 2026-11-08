```python
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items():
    save(k,v.astype(np.float32))
print(saved())
```