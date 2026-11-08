```python
s=open('/work/build4.py').read()
s=s.replace("""        j=n.u(5,'Terr',p); b[5][j]=-2*(d-1)    # Iso capture penalty
        for q in NB[p]: W[5][u('NO',q),j]+=1; W[5][u('S',q),j]-=1; W[5][u('O',q),j]+=2""","""        j=n.u(5,'Terr',p); b[5][j]=4-2*d    # Iso capture penalty z=2sO-2d+4-3sS-sNO
        for q in NB[p]: W[5][u('NO',q),j]-=1; W[5][u('S',q),j]-=3; W[5][u('O',q),j]+=2""")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4,U3=8,scap3=0.75,early=[-4,-2.5,0,0.3,2.5],**{'def':6})
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```