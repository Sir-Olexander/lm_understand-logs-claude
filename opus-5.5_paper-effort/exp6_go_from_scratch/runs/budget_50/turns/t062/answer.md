```python
s=open('/work/build2.py').read()
old=s[s.index("        for nm,th in (('U2',2),('U3',3)):"):s.index("        j=n.u(5,'Terr',p)")]
new="""        # U2=relu(1-A), U3=relu(2-A-2T); A=deg-sO+sS-2sUS+scap*sNO ; T=sum(US-NS)
        for nm,th,wU,wN in (('U2',1,2,0),('U3',2,0,2)):
            j=n.u(5,nm,p); b[5][j]=th-d
            for q in NB[p]:
                W[5][u('O',q),j]+=1; W[5][u('S',q),j]-=1; W[5][u('US',q),j]+=wU; W[5][u('NS',q),j]+=wN; W[5][u('NO',q),j]-=prm['scap']
            W[6][j,o]=-prm[nm]
"""
s=s.replace(old,new)
open('/work/build2.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build2
prm=build2.default_prm(); prm.update(U2=25,U3=25,**{'def':2})
n=build2.build(prm,inp,mv); n.save(save); print(n.used)
```