```python
import sys; sys.path.insert(0,'/work')
import importlib, build, pickle; importlib.reload(build)
P=pickle.load(open('/work/Pfinal_candidate.pkl','rb'))
W=build.build(P)
ok=all(np.array_equal(load(k),W[k].astype(np.float32)) for k in W)
print(saved(), ok)
import json
G=json.load(open('/work/pass_10/games.json'))
print(sum(1 for g in G if g['end']=='repetition' and g['result']==-1), 'repetition losses')
```