```python
import importlib, sys; sys.path.insert(0,'/work')
src=open('/work/v4_ab.py').read()
src=src.replace("U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*3.0)","U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*3.0-0.2)")
src=src.replace("U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*1.5)","U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*1.5-1.0)")
open('/work/v4_ab.py','w').write(src)
import v4_ab; importlib.reload(v4_ab)
P=dict(eps=1e-5,CAP=15.0,ESC=12.0,B1=4.0,BIG1=10.0,B0=24.0,BIG0=12.0,BIGC=100.0,SA1=5.0,SA0=1.5,EYE=40.0,C=10.0,
       LV={1:-2.0,2:-0.5,3:1.0,4:1.2,5:1.0},prox_own={1:0.2,2:0.2},prox_opp={1:0.6,2:0.3},PASS=-12.0)
net,U=v4_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v4_units.pkl','wb'))
```