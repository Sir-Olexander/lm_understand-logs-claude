```python
import sys; sys.path.insert(0,'/work')
import importlib, build
code=open('/work/build.py').read()
code=code.replace("a1=6.0,a2=3.0,a3=4.0,a4=2.0,","a1=9.0,a2=3.0,a3=6.0,a4=2.0,")
code=code.replace("mu_opp=1/3., d0_opp=3.0, mu_own=0.2, d0_own=5.0, Kk=3.0,","mu_opp=0.0, d0_opp=0.0, mu_own=0.0, d0_own=0.0, Kk=3.0,")
code=code.replace("sa_kill=1.5,","sa_kill=2.0,")
open('/work/build.py','w').write(code)
importlib.reload(build)
print({k:build.DEF[k] for k in ['a1','a2','a3','a4','mu_opp','mu_own','sa_kill','w_sa','w_cc']})
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
print(W['L5.W'][81+40,243+40], W['L5.W'][81+42,243+40], W['L5.W'][243+40,243+40])
```