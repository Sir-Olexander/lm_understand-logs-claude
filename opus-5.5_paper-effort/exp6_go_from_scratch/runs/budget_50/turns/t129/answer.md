```python
s=open('/work/build4.py').read()
a=s.index("        j=n.u(5,'Eye',p); b[5][j]=-2+(4-d)*prm['te_deg']"); bidx=s.index("        W[6][j,o]=-prm['iso']\n")+len("        W[6][j,o]=-prm['iso']\n")
s=s[:a]+"""        j=n.u(5,'Eye',p); b[5][j]=-d+1
        for m in NB[p]: W[5][u('S',m),j]+=1
        W[5][u('S',p),j]-=4; W[5][u('O',p),j]-=4
        W[6][j,o]=-prm['eye']
        j=n.u(5,'Terr',p); b[5][j]=-prm['terr_th']
        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2; W[5][u('NS',q),j]-=10
        for q in range(81):
            if D(p,q)==2: W[5][u('S',q),j]+=.5; W[5][u('O',q),j]-=1
        W[6][j,o]=-prm['terr']
"""+s[bidx:]
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
BASE=dict(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4,U3=8,scap3=0.5,pw2=0.6,early=[-4,-2.5,0,0.3,2.5],**{'def':6})
import json; json.dump(BASE,open('/work/base.json','w'))
prm=build4.default_prm(); prm.update(BASE)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```