```python
s=open('/work/build4.py').read()
s=s.replace("""    W[6][n.blk[5]['Score'],81]=prm['pw']""","""    W[6][n.blk[5]['Score'],81]=prm['pw']
    W[6][n.blk[5]['Ph1'],81]=prm['pph']; W[6][n.blk[5]['Ph2'],81]=-prm['pph']; b[6][81]-=50*prm['pph']""")
s=s.replace("""    W[6][n.blk[5]['Score'],81]=prm['pw']; b[6][81]=prm['pass']-100*prm['pw']""","""    b[6][81]=prm['pass']-100*prm['pw']""")
s=s.replace("komi=7.5,","komi=7.5,pph=0.3,")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
W6=n.W[6]; print(W6[n.blk[5]['Score'],81],W6[n.blk[5]['Ph1'],81],n.b[6][81])
```