```python
import re
s=open('/work/build.py').read()
s=s.replace("""            W[5][n.u(4,'NO',q),j]+=prm['cap']; W[5][n.u(4,'NS',q),j]+=prm['esc']""",
"""            W[5][n.u(4,'NO',q),j]+=prm['cap']""")
s=s.replace("""    b[6][81]=prm['pass']
    return n""","""    b[6][81]=prm['pass']
    # gated escape: E1=relu(sumNS + K(s-2)), E2=relu(K(s-2)); bonus esc*(E1-E2)
    n.alloc(5,'E1'); n.alloc(5,'E2'); K=4
    for p in range(81):
        for nm in ('E1','E2'):
            j=n.u(5,nm,p); b[5][j]=K*(len(NB[p])-2)
            for q in NB[p]:
                W[5][n.u(4,'O',q),j]-=K; W[5][n.u(4,'NS',q),j]-=K; W[5][n.u(4,'NO',q),j]+=2*K
                if nm=='E1': W[5][n.u(4,'NS',q),j]+=1
        W[6][n.u(5,'E1',p),mv(p//9,p%9)]=prm['esc']; W[6][n.u(5,'E2',p),mv(p//9,p%9)]=-prm['esc']
    # influence bump: c(m)=relu(I+th)-2relu(I)+relu(I-th), stones masked; value(p)=sum K(p,m)c(m)
    th=prm['bth']; ik=prm['ikern']; vk=prm['vkern']
    D=lambda a,c: abs(a//9-c//9)+abs(a%9-c%9)
    for k,nm,off,sg in ((3,'B1',th,1),(4,'B2',0,-2),(5,'B3',-th,1)):
        n.alloc(k,nm)
        for m in range(81):
            j=n.u(k,nm,m); b[k][j]=off
            for q in range(81):
                d=D(m,q)
                if 1<=d<len(ik):
                    W[k][n.u(k-1,'S',q),j]+=ik[d]; W[k][n.u(k-1,'O',q),j]-=ik[d]
            W[k][n.u(k-1,'S',m),j]-=100; W[k][n.u(k-1,'O',m),j]-=100
            for p in range(81):
                d=D(m,p)
                if d<len(vk):
                    tgt=n.u(k+1,'acc',p) if k<5 else mv(p//9,p%9)
                    (W[k+1] if k<5 else W[6])[j,tgt]+=sg*vk[d]*prm['bw']
    return n""")
s=s.replace("cap=8,esc=5,uns=5,**{'pass':-1.0})","cap=8,esc=5,uns=8,bth=2.0,ikern=[0,1,.5,.25],vkern=[1,.5,.25],bw=0.4,**{'pass':-2.0})")
open('/work/build.py','w').write(s)
import sys; sys.path.insert(0,'/work')
import build
n=build.build(build.default_prm(),inp,mv)
print(n.used); n.save(save)
```