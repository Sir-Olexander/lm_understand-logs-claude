```python
s=open('/work/build2.py').read()
s=s.replace("""        W[5][u('S',p),j]-=4; W[5][u('O',p),j]-=4
        W[6][j,o]=-prm['eye']""","""        W[5][u('S',p),j]-=4; W[5][u('O',p),j]-=4
        for m in NB[p]: W[5][u('NS',m),j]-=4
        W[6][j,o]=-prm['eye']""")
s=s.replace("""        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2
""","""        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2; W[5][u('NS',q),j]-=10
""")
open('/work/build2.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build2
prm=build2.default_prm(); prm.update(scap=2)
n=build2.build(prm,inp,mv); n.save(save); print(n.used)
```