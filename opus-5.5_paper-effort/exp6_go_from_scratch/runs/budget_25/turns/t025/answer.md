```python
import re
src=open('/work/v2_ab.py').read()
# U1/U0 capture suppression and new U0 constants, DEF term
src=src.replace("""        U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*1.5)""","""        t1+=[(U[('n',5,q)],-P['BIGC']) for q in NB[p]]
        U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*1.5)""")
src=src.replace("""        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*0.5)""","""        t0+=[(U[('n',5,q)],-P['BIGC']) for q in NB[p]]
        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*0.5)""")
open('/work/v2_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v2_ab; importlib.reload(v2_ab)
P=dict(eps=1e-4,CAP=15.0,ESC=12.0,B1=8.0,BIG1=20.0,B0=24.0,BIG0=12.0,BIGC=100.0,SA1=5.0,SA0=1.5,EYE=40.0,
       LV={1:-2.0,2:-0.5,3:1.0,4:1.2,5:1.0},prox_own={1:0.2,2:0.2},prox_opp={1:0.6,2:0.3},PASS=-12.0)
net,U=v2_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v2_units.pkl','wb'))
print('U1' in src, src.count('BIGC'))
```