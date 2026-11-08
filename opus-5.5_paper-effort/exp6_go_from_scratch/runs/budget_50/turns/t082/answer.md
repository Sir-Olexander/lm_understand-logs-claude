```python
s=open('/work/build4.py').read()
s=s.replace("""    b[6][81]=prm['pass']
    return n""","""    b[6][81]=prm['pass']
    # score estimate -> pass logit
    for nm in ('TS','TO'): n.alloc(0,nm)
    jk=n.alloc(0,'K',1); W[0][inp(0,0,2),jk]=1
    for p in range(81):
        for nm,a,bb in (('TS',s_,o_),('TO',o_,s_)):
            j=n.u(0,nm,p)
            for m in NB[p]: W[0][a(m),j]+=1; W[0][bb(m),j]-=4
            W[0][s_(p),j]-=4; W[0][o_(p),j]-=4
    js=n.alloc(1,'Score',1); b[1][js]=100-prm['komi']; W[1][jk,js]=2*prm['komi']
    for p in range(81):
        W[1][n.u(0,'S',p),js]+=1; W[1][n.u(0,'O',p),js]-=1
        W[1][n.u(0,'TS',p),js]+=prm['tw']; W[1][n.u(0,'TO',p),js]-=prm['tw']
    for k in (2,3,4,5):
        j=n.alloc(k,'Score',1); W[k][n.blk[k-1]['Score'],j]=1
    W[6][n.blk[5]['Score'],81]=prm['pw']; b[6][81]=prm['pass']-100*prm['pw']
    return n""")
s=s.replace("**{'pass':-4.0})","komi=7.5,tw=0.6,pw=0.4,**{'pass':-4.0})")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
n=build4.build(build4.default_prm(),inp,mv); n.save(save); print(n.used)
```