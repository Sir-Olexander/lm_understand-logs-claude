```python
src=open('/work/v16_ab.py').read()
old="""        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))"""
new="""        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))
        if p in P['UXPTS']:
            tx=[(U[('O',5,p)],-1.0/eps)]
            for q in NB[p]:
                for k in own_colors(q): tx.append((U[('m',5,q,k)],1.0))
            U[('Ux',6,p)]=net.unit(6,('Ux',p),tx,1.0-0.5-0.1)"""
assert old in src; src=src.replace(old,new)
old="""        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]"""
new="""        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]
        if ('Ux',6,p) in U: terms.append((U[('Ux',6,p)],-P['WX']))"""
assert old in src; src=src.replace(old,new)
open('/work/v18_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v18_ab; importlib.reload(v18_ab)
UX=[(0,0),(0,8),(8,0),(8,8),(0,1),(1,0),(0,7),(1,8),(7,0),(8,1),(7,8),(8,7),(0,2),(2,0),(0,6),(2,8),(6,0),(8,2),(8,6),(6,8),(0,4),(8,4)]
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=40.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.2,kD=0.3,Scap=20.0,Dcap=10.0,PASS=-12.0,cert_cs=[0,2,3,4,6],
       UXPTS=set(r*9+c for r,c in UX),WX=60.0)
net,U=v18_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v18_units.pkl','wb'))
```