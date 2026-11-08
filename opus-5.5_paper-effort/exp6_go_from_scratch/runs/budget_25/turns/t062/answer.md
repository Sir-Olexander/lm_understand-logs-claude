```python
src=open('/work/v5_ab.py').read()
old="""        U[('P1',6,p)]=net.unit(6,('P1',p),t1,P['C']+(lam-mu)*axb)"""
new="""        t1.append((U[('O',5,p)],P['eta']/eps))
        U[('P1',6,p)]=net.unit(6,('P1',p),t1,P['C']+(lam-mu)*axb-P['eta'])"""
assert old in src; src=src.replace(old,new)
old="""        net.outw(p,terms,-P['C']+P['LV'][line(p)])"""
new="""        terms.append((U[('gS',6)],P['kL'].get(line(p),0.0)))
        net.outw(p,terms,-P['C']+P['LV'][line(p)])"""
assert old in src; src=src.replace(old,new)
open('/work/v6_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v6_ab; importlib.reload(v6_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,B1=4.0,BIG1=10.0,B0=24.0,BIG0=12.0,BIGC=100.0,SA1=5.0,SA0=1.5,EYEU=80.0,C=30.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.4,kD=0.5,PASS=-12.0)
net,U=v6_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v6_units.pkl','wb'))
```