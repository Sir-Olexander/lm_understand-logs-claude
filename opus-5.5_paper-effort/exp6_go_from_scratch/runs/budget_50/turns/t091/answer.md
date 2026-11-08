```python
s=open('/work/build4.py').read()
s=s.replace("""    b[6][81]=prm['pass']-100*prm['pw']""","""    b[6][81]=prm['pass']-100*prm['pw']-50*prm['pph']""")
s=s.replace("b[6][81]-=50*prm['pph']","")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4)
n=build4.build(prm,inp,mv); n.save(save); print(n.b[6][81])
```