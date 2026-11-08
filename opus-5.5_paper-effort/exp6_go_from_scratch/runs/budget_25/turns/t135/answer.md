```python
src=open('/work/v18_ab.py').read()
old="""        t2=[(i,w) for i,w in Ax]+[(i,-w) for i,w in Ao]
        U[('P2',6,p)]=net.unit(6,('P2',p),t2,axb)"""
new="""        t2=[(i,w) for i,w in Ax]+[(i,-w) for i,w in Ao]
        if p not in P['NOP2']:
            U[('P2',6,p)]=net.unit(6,('P2',p),t2,axb)"""
assert old in src; src=src.replace(old,new)
old="""        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]"""
new="""        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0)]
        if ('P2',6,p) in U: terms.append((U[('P2',6,p)],-P['lam']))"""
assert old in src; src=src.replace(old,new)
open('/work/v19_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v19_ab; importlib.reload(v19_ab)
def line(s):
    r,c=divmod(s,9); return min(r,c,8-r,8-c)+1
NOP2=[(0,3),(0,5),(3,0),(5,0),(3,8),(5,8),(8,3),(8,5),(4,0),(4,8)]
UXPTS=set(s for s in range(81) if line(s)==1)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=40.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.2,kD=0.3,Scap=20.0,Dcap=10.0,PASS=-12.0,cert_cs=[0,2,3,4,6],
       UXPTS=UXPTS,WX=150.0,NOP2=set(r*9+c for r,c in NOP2))
net,U=v19_ab.build(P)
print(net.n, len(UXPTS))
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v19_units.pkl','wb'))
```