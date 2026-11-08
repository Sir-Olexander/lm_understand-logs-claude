```python
src=open('/work/v2_ab.py').read()
old="""        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))"""
new="""        te=[(U[('O',5,n)],1.0) for n in NB[p]]
        for q in NB[p]:
            for k in own_colors(q): te.append((U[('m',5,q,k)],-1.0))
        U[('eye',6,p)]=net.unit(6,('eye',p),te,-(deg-1))"""
assert old in src
src=src.replace(old,new)
open('/work/v3_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v3_ab; importlib.reload(v3_ab)
P=dict(eps=1e-4,CAP=15.0,ESC=20.0,B1=8.0,BIG1=20.0,B0=24.0,BIG0=12.0,BIGC=100.0,SA1=6.0,SA0=1.0,EYE=40.0,
       LV={1:-1.5,2:-0.3,3:0.8,4:1.0,5:0.8},prox_own={1:0.0,2:0.3},prox_opp={1:0.4,2:0.4},PASS=-12.0)
net,U=v3_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v3_units.pkl','wb'))
```