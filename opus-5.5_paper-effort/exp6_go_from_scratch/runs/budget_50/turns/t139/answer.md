```python
s=open('/work/build4.py').read()
s=s.replace("""        if prm['iso5']: W[5][u('Iso',p),j]-=prm['iso']
            W[5][u('O',q),j]+=prm['c_opp']; W[5][u('S',q),j]+=prm['c_own']""","""            W[5][u('O',q),j]+=prm['c_opp']; W[5][u('S',q),j]+=prm['c_own']
        if prm['iso5']: W[5][u('Iso',p),j]-=prm['iso']""")
open('/work/build4.py','w').write(s)
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json'))
prm=build4.default_prm(); prm.update(BASE); prm.update(iso5=True,iso=6)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```