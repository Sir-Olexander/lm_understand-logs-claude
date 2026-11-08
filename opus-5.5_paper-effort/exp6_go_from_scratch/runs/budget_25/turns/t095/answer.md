```python
import importlib, sys; sys.path.insert(0,'/work')
import v9_ab; importlib.reload(v9_ab)
s=1.0
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=20.0,EYE=40.0,C=40.0,
       LV={1:-6.0*s,2:-1.5*s,3:1.5*s,4:2.25*s,5:3.0*s},kern={1:1.0,2:0.6,3:0.3},lam=4.5*s,mu=0.75*s,eta=0.4*s,
       kL={1:0.3*s,2:0.075*s},S0=50.0,D0=10.0,kS=0.4,kD=0.5,Scap=20.0,Dcap=10.0,PASS=-12.0)
net,U=v9_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v11_units.pkl','wb'))
```