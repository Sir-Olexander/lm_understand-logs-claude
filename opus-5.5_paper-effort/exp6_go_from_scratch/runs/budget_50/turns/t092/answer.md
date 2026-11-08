```python
s=open('/work/build4.py').read()
s=s.replace("W[6][n.blk[5]['Ph2'],81]=-prm['pph']; ; b[6][81]=prm['pass']-100*prm['pw']","W[6][n.blk[5]['Ph2'],81]=-prm['pph']")
open('/work/build4.py','w').write(s)
i=s.index("score estimate"); print(s[i-200:i+100]); print(s[s.index("Ph2'],81]")-150:s.index("Ph2'],81]")+120])
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4)
n=build4.build(prm,inp,mv); n.save(save); print(n.b[6][81])
```