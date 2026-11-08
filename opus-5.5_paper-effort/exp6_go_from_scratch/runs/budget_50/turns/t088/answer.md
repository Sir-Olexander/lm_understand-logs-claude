```python
s=open('/work/build4.py').read()
s=s.replace("""    W[6][n.blk[5]['Score'],81]=prm['pw']""","""    # phase: stones on board, clamped at 50 -> per-point prior shift
    j1=n.alloc(1,'Ph1',1); j2=n.alloc(1,'Ph2',1); b[1][j2]=-50
    for p in range(81):
        for jj in (j1,j2): W[1][n.u(0,'S',p),jj]=1; W[1][n.u(0,'O',p),jj]=1
    for k in (2,3,4,5):
        for nm in ('Ph1','Ph2'):
            j=n.alloc(k,nm,1); W[k][n.blk[k-1][nm],j]=1
    for p in range(81):
        r,c=divmod(p,9); line=min(r,c,8-r,8-c); o=mv(r,c)
        early=prm['early'][line]; late=prm['late'][line]
        b[6][o]+=early; W[6][n.blk[5]['Ph1'],o]=(late-early)/50; W[6][n.blk[5]['Ph2'],o]=-(late-early)/50
    W[6][n.blk[5]['Score'],81]=prm['pw']""")
s=s.replace("komi=7.5,","komi=7.5,early=[-3,-1.2,0,0,0],late=[-0.5,0,0,0,0],")
s=s.replace("pr[p]=[-1.5,-0.5,0,0.2,0.4][line]","pr[p]=[0,0,0,0.2,0.4][line]")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
prm=build4.default_prm(); prm.update(c_opp=1.0,c_own=0.5,c_dg=0.4)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```