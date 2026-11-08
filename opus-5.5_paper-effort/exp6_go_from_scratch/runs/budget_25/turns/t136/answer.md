```python
src=open('/work/v19_ab.py').read()
old="""        lam,mu=P['lam'],P['mu']
        t1=[(i,(lam-mu)*w) for i,w in Ax]+[(i,-mu*w) for i,w in Ao]"""
new="""        lam,mu=P['lam'],P['mu']
        if p in P['NOP2']: lam=0.0
        t1=[(i,(lam-mu)*w) for i,w in Ax]+[(i,-mu*w) for i,w in Ao]"""
assert old in src; src=src.replace(old,new)
open('/work/v19_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v19_ab; importlib.reload(v19_ab)
import pickle
P,_=pickle.load(open('/work/v19_units.pkl','rb'))
net,U=v19_ab.build(P)
print(net.n)
# check bias of P1 at a NOP2 point vs normal
import numpy as np
p=3; j=U[('P1',6,p)]; print('P1 bias at (0,3)',net.B[5][j], 'P2 exists', ('P2',6,p) in U, 'Ux exists',('Ux',6,p) in U)
p=1; j=U[('P1',6,p)]; print('P1 bias at (0,1)',net.B[5][j], ('P2',6,p) in U, ('Ux',6,p) in U)
net.save_all(save)
pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v19_units.pkl','wb'))
```