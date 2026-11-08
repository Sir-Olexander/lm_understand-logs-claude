```python
import importlib, sys; sys.path.insert(0,'/work')
import v18_ab; importlib.reload(v18_ab)
UX=[(0,0),(0,8),(8,0),(8,8),(0,1),(1,0),(0,7),(1,8),(7,0),(8,1),(7,8),(8,7),(0,2),(2,0),(0,6),(2,8),(6,0),(8,2),(8,6),(6,8),(0,4),(8,4)]
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=40.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.2,kD=0.3,Scap=20.0,Dcap=10.0,PASS=-12.0,cert_cs=[0,2,3,4,6],
       UXPTS=set(r*9+c for r,c in UX),WX=100.0)
net,U=v18_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v18_units.pkl','wb'))
```