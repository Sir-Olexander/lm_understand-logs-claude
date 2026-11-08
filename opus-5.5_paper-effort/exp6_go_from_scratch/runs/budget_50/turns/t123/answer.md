```python
s=open('/work/build4.py').read()
a=s.index("        j=n.u(5,'Eye',p); b[5][j]=-d+1"); bidx=s.index("        W[6][j,o]=-prm['eye']")+len("        W[6][j,o]=-prm['eye']\n")
s=s[:a]+"""        j=n.u(5,'Eye',p); b[5][j]=-2+(4-d)*prm['te_deg']   # merged eye/territory unit
        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2
        for q in range(81):
            if D(p,q)==2: W[5][u('S',q),j]+=.5; W[5][u('O',q),j]-=1
        W[5][u('S',p),j]-=6; W[5][u('O',p),j]-=6
        W[6][j,o]=-prm['te']
        j=n.u(5,'Terr',p); b[5][j]=-2*(d-1)    # Iso capture penalty
        for q in NB[p]: W[5][u('NO',q),j]+=1; W[5][u('S',q),j]-=1; W[5][u('O',q),j]+=2
        W[5][u('S',p),j]-=10; W[5][u('O',p),j]-=10
        W[6][j,o]=-prm['iso']
"""+s[bidx:]
a=s.index("        j=n.u(5,'Terr',p); b[5][j]=-prm['terr_th']"); bidx=s.index("        W[6][j,o]=-prm['terr']\n")+len("        W[6][j,o]=-prm['terr']\n")
s=s[:a]+s[bidx:]
s=s.replace("komi=7.5,","komi=7.5,te=2.5,te_deg=1.0,iso=6,")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import importlib, build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4,U3=8,scap3=0.75,early=[-4,-2.5,0,0.3,2.5],**{'def':6})
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```