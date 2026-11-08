```python
src=open('/work/v8_ab.py').read()
old="""            terms=[(U[('x',4,t,k)],-6.0) for t in [q]+NB[q]]+[(U[('b',4,q)],-10.0)]
            terms+=[(U[('b',4,t)],1.0) for t in NB[q]]
            U[('n4c',5,q)]=net.unit(5,('n4c',q),terms,1.0-len(NB[q])-1.0)"""
new="""            terms=[(U[('x',4,t,k)],-6.0) for t in [q]+NB[q]]+[(U[('b',4,q)],-10.0)]
            terms+=[(U[('x',4,t,1-k)],-P['Vg']) for t in [q]+NB[q]]
            U[('n4c',5,q)]=net.unit(5,('n4c',q),terms,1.0)"""
assert old in src; src=src.replace(old,new)
open('/work/v12_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v12_ab; importlib.reload(v12_ab)
P=dict(eps=1e-5,CAP=25.0,ESC=16.0,Ua=2.0,Ub=1.0,Uw=1.0,Uk=1.0,BIGC=100.0,SAc=20.0,EYE=40.0,C=40.0,Vg=1/25.,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.4,kD=0.5,Scap=20.0,Dcap=10.0,PASS=-12.0)
net,U=v12_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v12_units.pkl','wb'))
print([k for k in U if k[0]=='n4c'][:3], len([k for k in U if k[0]=='n4c']))
```