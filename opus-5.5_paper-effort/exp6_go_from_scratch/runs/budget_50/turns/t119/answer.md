```python
s=open('/work/build4.py').read()
s=s.replace("""    b[6][81]=prm['pass']-100*prm['pw']-50*prm['pph']""","""    b[6][81]=prm['pass']-100*prm['pw']-50*prm['pph']
    for nm,th,w in (('Sc25',125,-prm['pw']),('Sc10',100+prm['pth1'],prm['pw2']),('Sc25b',125,-prm['pw2'])):
        j=n.alloc(5,nm,1); W[5][n.blk[4]['Score'],j]=1; b[5][j]=-th; W[6][j,81]=w""")
s=s.replace("komi=7.5,","komi=7.5,pth1=10,pw2=0.4,")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4,pw=0.25,tw=0.4,U3=8,scap3=0.75,early=[-4,-2.5,0,0.3,2.5],**{'def':6})
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```