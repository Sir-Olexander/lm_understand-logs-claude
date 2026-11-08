```python
s=open('/work/build2.py').read()
# (a) LS-only certificate: Ev = relu(2L(q)+sum L(n) - 2 - 20(1-col))
s=s.replace("""            W[1][n.u(0,'L'+col+'2',p),j]+=-1; W[1][n.u(0,'L'+col,p),j]+=2
            for m in NB[p]:
                W[1][n.u(0,'L'+col,m),j]+=1; W[1][n.u(0,'L'+col+'2',m),j]-=1""","""            W[1][n.u(0,'L'+col,p),j]+=2
            for m in NB[p]: W[1][n.u(0,'L'+col,m),j]+=1""")
s=s.replace("""            for nm,off in (('L'+col,0),('L'+col+'2',1)):""","""            for nm,off in (('L'+col,0),):""")
s=s.replace("for nm in ['S','O','LS','LS2','LO','LO2']: n.alloc(0,nm)","for nm in ['S','O','LS','LO']: n.alloc(0,nm)")
# (b) isolated capture penalty unit: z = sumNO - 2 sumS(n) - 2(#empty-1)
s=s.replace("""    b[6][81]=prm['pass']""","""        j=n.alloc(5,'Iso_%d'%p,1); d=len(NB[p]); b[5][j]=-2*(d-1)
        for q in NB[p]:
            W[5][u('NO',q),j]+=1; W[5][u('S',q),j]+=2-2; W[5][u('O',q),j]+=2
        W[6][j,o]=-prm['iso']
    b[6][81]=prm['pass']""")
s=s.replace("scap=2,","scap=2,iso=6,")
open('/work/build2.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build2
prm=build2.default_prm()
n=build2.build(prm,inp,mv); n.save(save); print(n.used)
```