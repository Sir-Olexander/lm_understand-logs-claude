```python
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
import pickle; pickle.dump(dict(build.DEF),open('/work/Pfinal_candidate.pkl','wb'))
print({k:build.DEF[k] for k in ['seed','tengen','a1','a2','a3','a4','sa_kill','w_sa','w_cc','alpha','mu_opp']})
```