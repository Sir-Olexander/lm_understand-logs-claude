```python
import importlib, sys; sys.path.insert(0,'/work')
import v9_ab; importlib.reload(v9_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=40.0,
       LV={1:-6.0,2:-2.0,3:1.0,4:2.25,5:3.5},kern={1:1.0,2:0.6,3:0.3},lam=6.0,mu=0.4,eta=0.4,
       kL={1:0.3,2:0.1},S0=50.0,D0=10.0,kS=0.2,kD=0.3,Scap=20.0,Dcap=10.0,PASS=-12.0)
net,U=v9_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v15_units.pkl','wb'))
```