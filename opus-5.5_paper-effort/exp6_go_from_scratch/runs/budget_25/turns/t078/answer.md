```python
src=open('/work/v7_ab.py').read()
# L5 changes: b5 for all points; replace n4 with n4c (subset); v5 unchanged
old="""    for s in BSUB:
        U[('b',5,s)]=net.unit(5,('b',s),[(U[('b',4,s)],1.0)],0.0)"""
new="""    for s in range(81):
        U[('b',5,s)]=net.unit(5,('b',s),[(U[('b',4,s)],1.0)],0.0)"""
assert old in src; src=src.replace(old,new)
old="""        k=par(q)
        terms=[(U[('x',4,t,k)],-2.0) for t in [q]+NB[q]]
        U[('n4',5,q)]=net.unit(5,('n4',q),terms+[(U[('b',4,q)],-5.0)],1.0)"""
new="""        if q in BSUB:
            k=par(q)
            terms=[(U[('x',4,t,k)],-6.0) for t in [q]+NB[q]]+[(U[('b',4,q)],-10.0)]
            terms+=[(U[('b',4,t)],1.0) for t in NB[q]]
            U[('n4c',5,q)]=net.unit(5,('n4c',q),terms,1.0-len(NB[q])-1.0)"""
assert old in src; src=src.replace(old,new)
# L6 n5 via P/E formula
old="""        terms=[(U[('n4',5,q)],2.0)]+[(U[('v',5,t)],-2.0) for t in NB[q]]
        U[('n5',6,q)]=net.unit(6,('n5',q),terms,-1.0)"""
new="""        BIGP=10.0
        # P_q = 1 + O5'_q - b5_q ; n_e(q) = sum_t (b5_t - 2 O5'_t)
        terms=[(U[('O',5,q)],BIGP),(U[('b',5,q)],-BIGP)]
        terms+=[(U[('v',5,t)],-2.0) for t in NB[q]]
        terms+=[(U[('b',5,t)],-1.0) for t in NB[q]]+[(U[('O',5,t)],2.0) for t in NB[q]]
        U[('n5',6,q)]=net.unit(6,('n5',q),terms,1.0+1.0)"""
assert old in src; src=src.replace(old,new)
old="""            tu.append((U[('n4',5,q)],-P['BIGC']))"""
new="""            if ('n4c',5,q) in U: tu.append((U[('n4c',5,q)],-P['BIGC']))"""
assert old in src; src=src.replace(old,new)
# positional: all points have b5 now
src=src.replace("""            if t in BSUB:
                Ax+=[(U[('O',5,t)],w),(U[('b',5,t)],-w)]; axb+=w""","""            Ax+=[(U[('O',5,t)],w),(U[('b',5,t)],-w)]; axb+=w""")
open('/work/v8_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v8_ab; importlib.reload(v8_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=30.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.4,kD=0.5,PASS=-12.0)
net,U=v8_ab.build(P)
print(net.n)
```