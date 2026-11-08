```python
src=open('/work/v6_ab.py').read()
i0=src.index("        # n_e = O5'_p/eps - 1"); i1=src.index("        # positional: A_o, A_x")
newU='''        # combined self-atari unit Uc = relu(a - b*n_e - c2*sum W - c1*k - BIGC*sum n4)
        a,b_,c2,c1=P['Ua'],P['Ub'],P['Uw'],P['Uk']
        tu=[(U[('O',5,p)],-b_/eps)]; bias=a+b_
        for q in NB[p]:
            # W_q = O_q - sum m ; k = sum O_q
            tu.append((U[('O',5,q)],-c2-c1))
            for k in own_colors(q): tu.append((U[('m',5,q,k)],c2))
            tu.append((U[('n4',5,q)],-P['BIGC']))
        U[('Uc',6,p)]=net.unit(6,('Uc',p),tu,bias-0.3)
        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))
'''
src=src[:i0]+newU+src[i1:]
old="""        terms+=[(U[('U1',6,p)],-P['SA1']),(U[('U0',6,p)],-P['SA0']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]"""
new="""        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]"""
assert old in src; src=src.replace(old,new)
open('/work/v7_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v7_ab; importlib.reload(v7_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=30.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.4,kD=0.5,PASS=-12.0)
net,U=v7_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v7_units.pkl','wb'))
```