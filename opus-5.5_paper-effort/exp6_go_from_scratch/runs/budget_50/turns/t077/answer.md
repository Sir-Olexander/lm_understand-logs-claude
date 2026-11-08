```python
s=open('/work/build4.py').read()
s=s.replace("""        for nm,c,al,be in (('U2',1,.9,.3),('U3',2.5,1,.5)):""","""        for nm,c,al,be,sc in (('U2',1,.9,.3,prm['scap']),('U3',2.5,1,.5,prm['scap3'])):""")
s=s.replace("W[5][u('NO',q),j]-=al*prm['scap']","W[5][u('NO',q),j]-=al*sc")
s=s.replace("scap=2,","scap=2,scap3=0,")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
n=build4.build(build4.default_prm(),inp,mv); n.save(save); print(n.used)
```