```python
src=open('/work/v8_ab.py').read()
src=src.replace("        if q in BSUB:\n            k=par(q)","        if q in BSUB and q not in (16,64):\n            k=par(q)")
old="""    U[('gS',1)]=net.unit(1,'gS',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0'])
    U[('gD',1)]=net.unit(1,'gD',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0'])
    for l in (2,3,4,5):
        for g in ('gS','gD'):"""
new="""    U[('gS',1)]=net.unit(1,'gS',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0'])
    U[('gD',1)]=net.unit(1,'gD',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0'])
    U[('gS2',1)]=net.unit(1,'gS2',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0']-P['Scap'])
    U[('gD2',1)]=net.unit(1,'gD2',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0']-P['Dcap'])
    for l in (2,3,4,5,6):
        for g in ('gS2','gD2'):
            U[(g,l)]=net.unit(l,g,[(U[(g,l-1)],1.0)],0.0)
    for l in (2,3,4,5):
        for g in ('gS','gD'):"""
assert old in src; src=src.replace(old,new)
old="""    net.outw(81,[(U[('gS',6)],P['kS']),(U[('gD',6)],P['kD'])],P['PASS'])"""
new="""    net.outw(81,[(U[('gS',6)],P['kS']),(U[('gD',6)],P['kD']),(U[('gS2',6)],-P['kS']),(U[('gD2',6)],-P['kD'])],P['PASS'])"""
assert old in src; src=src.replace(old,new)
open('/work/v8_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v8_ab; importlib.reload(v8_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=30.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.4,kD=0.5,Scap=20.0,Dcap=10.0,PASS=-12.0)
net,U=v8_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v8_units.pkl','wb'))
```