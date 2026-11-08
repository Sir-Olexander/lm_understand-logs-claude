```python
src=open('/work/v9_ab.py').read()
old="""    # ---- L2..L4"""
new="""    # edge certification units (opp): three opp stones in a row on 2nd line, end stones' 1st-line points empty
    CERT=[]
    for c in P['cert_cs']:
        for (a,b2,l1,l2) in [((1,c),(1,c+1),(0,c),(0,c+2)),((7,c),(7,c+1),(8,c),(8,c+2)),((c,1),(c+1,1),(c,0),(c+2,0)),((c,7),(c+1,7),(c,8),(c+2,8))]:
            pass
    for c in P['cert_cs']:
        trip=[[(1,c),(1,c+1),(1,c+2),(0,c),(0,c+2)],[(7,c),(7,c+1),(7,c+2),(8,c),(8,c+2)],[(c,1),(c+1,1),(c+2,1),(c,0),(c+2,0)],[(c,7),(c+1,7),(c+2,7),(c,8),(c+2,8)]]
        for t in trip:
            st=[r*9+cc for r,cc in t[:3]]; lb=[r*9+cc for r,cc in t[3:]]
            terms=[(IP(s),3.0) for s in st]+[(IO(x),-1.0) for x in lb]+[(IP(x),-1.0) for x in lb]
            j=net.unit(1,('cert',st[1]),terms,2.0-10.0)
            CERT.append((st[1],par(st[0]),j))
    # ---- L2..L4"""
assert old in src; src=src.replace(old,new)
old="""            for k in (0,1):
                terms=[(U[('x',l-1,t,k)],1.0) for t in [s]+NB[s]]
                U[('x',l,s,k)]=net.unit(l,('x',s,k),terms+[(U[('b',l-1,s)],-M)],0.0)"""
new="""            for k in (0,1):
                terms=[(U[('x',l-1,t,k)],1.0) for t in [s]+NB[s]]
                if l==2: terms+=[(j,1.0) for (sm,kk,j) in CERT if sm==s and kk==k]
                U[('x',l,s,k)]=net.unit(l,('x',s,k),terms+[(U[('b',l-1,s)],-M)],0.0)"""
assert old in src; src=src.replace(old,new)
open('/work/v16_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v16_ab; importlib.reload(v16_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,Ua=4.0,Ub=3.0,Uw=2.5,Uk=1.5,BIGC=100.0,SAc=12.0,EYE=40.0,C=40.0,
       LV={1:-6.0,2:-1.5,3:1.5,4:2.25,5:3.0},kern={1:1.0,2:0.6,3:0.3},lam=4.5,mu=0.75,eta=0.4,
       kL={1:0.3,2:0.075},S0=50.0,D0=10.0,kS=0.2,kD=0.3,Scap=20.0,Dcap=10.0,PASS=-12.0,cert_cs=[0,2,3,4,6])
net,U=v16_ab.build(P)
print(net.n)
```