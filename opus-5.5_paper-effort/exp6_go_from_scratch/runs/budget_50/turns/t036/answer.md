```python
import sys; sys.path.insert(0,'/work')
s=open('/work/build2.py').read()
s=s.replace("W[5][u('NS',q),j]+=1; W[5][u('NO',q),j]-=2","W[5][u('NS',q),j]+=1; W[5][u('NO',q),j]-=prm['scap']")
s=s.replace("U2=25,U3=3,","U2=25,U3=3,scap=2,")
open('/work/build2.py','w').write(s)
import build2
prm=build2.default_prm(); prm.update(scap=1,U3=6,**{'def':7})
n=build2.build(prm,inp,mv); n.save(save); print(n.used)
```