```python
code = r'''
import numpy as np, sys
sys.path.insert(0,'/work')
from netlib import *
def build(prm, inp, mv):
    n=Net(); W=n.W; b=n.b; C=100.0
    s_=lambda p: inp(p//9,p%9,0); o_=lambda p: inp(p//9,p%9,1)
    # L1
    for nm in ['S','O','LS','LS2','LO','LO2']: n.alloc(0,nm)
    for p in range(81):
        W[0][s_(p),n.u(0,'S',p)]=1; W[0][o_(p),n.u(0,'O',p)]=1
        d=len(NB[p])
        for col,pl in (('S',s_),('O',o_)):
            for nm,off in (('L'+col,0),('L'+col+'2',1)):
                j=n.u(0,nm,p)
                for m in NB[p]: W[0][s_(m),j]-=1; W[0][o_(m),j]-=1
                W[0][pl(p),j]+=4; b[0][j]=d-4-off
    # L2
    for nm in ['S','O','EvS','EvO','Eye','acc']: n.alloc(1,nm)
    for p in range(81):
        for nm in ['S','O']: W[1][n.u(0,nm,p),n.u(1,nm,p)]=1
        for col in 'SO':
            j=n.u(1,'Ev'+col,p)
            W[1][n.u(0,'L'+col+'2',p),j]+=-1; W[1][n.u(0,'L'+col,p),j]+=2
            for m in NB[p]:
                W[1][n.u(0,'L'+col,m),j]+=1; W[1][n.u(0,'L'+col+'2',m),j]-=1
            W[1][n.u(0,col,p),j]+=20; b[1][j]=-22
        j=n.u(1,'Eye',p); d=len(NB[p])
        for m in NB[p]: W[1][n.u(0,'S',m),j]+=1
        W[1][n.u(0,'S',p),j]-=4; W[1][n.u(0,'O',p),j]-=4; b[1][j]=-d+1
        j=n.u(1,'acc',p); b[1][j]=C+prm['prior'][p]
        for q in NB[p]:
            W[1][n.u(0,'O',q),j]+=2*prm['att']; W[1][n.u(0,'LO',q),j]-=prm['att']
    # L3
    for nm in ['S','O','NS','NO','Terr','acc']: n.alloc(2,nm)
    for p in range(81):
        for nm in ['S','O']: W[2][n.u(1,nm,p),n.u(2,nm,p)]=1
        for col in 'SO':
            j=n.u(2,'N'+col,p); W[2][n.u(1,col,p),j]=1
            for m in [p]+NB[p]: W[2][n.u(1,'Ev'+col,m),j]-=1
        j=n.u(2,'Terr',p)
        for m in NB[p]: W[2][n.u(1,'S',m),j]+=1; W[2][n.u(1,'O',m),j]-=2
        for m in R2[p]: W[2][n.u(1,'S',m),j]+=0.5; W[2][n.u(1,'O',m),j]-=1
        b[2][j]=-prm['terr_th']
        j=n.u(2,'acc',p); W[2][n.u(1,'acc',p),j]=1; W[2][n.u(1,'Eye',p),j]=-prm['eye']
    # L4, L5
    for k in (3,4):
        for nm in ['S','O','NS','NO','acc']: n.alloc(k,nm)
        for p in range(81):
            for nm in ['S','O','acc']: W[k][n.u(k-1,nm,p),n.u(k,nm,p)]=1
            if k==3: W[k][n.u(2,'Terr',p),n.u(k,'acc',p)]=-prm['terr']
            for col in 'SO':
                j=n.u(k,'N'+col,p)
                for m in [p]+NB[p]:
                    W[k][n.u(k-1,col,m),j]-=1; W[k][n.u(k-1,'N'+col,m),j]+=1
                W[k][n.u(k-1,col,p),j]+=1
    # L6
    for nm in ['acc','Uns']: n.alloc(5,nm)
    for p in range(81):
        j=n.u(5,'acc',p); W[5][n.u(4,'acc',p),j]=1
        for q in NB[p]:
            W[5][n.u(4,'NO',q),j]+=prm['cap']; W[5][n.u(4,'NS',q),j]+=prm['esc']
        j=n.u(5,'Uns',p); b[5][j]=2-len(NB[p])
        for q in NB[p]:
            W[5][n.u(4,'O',q),j]+=1; W[5][n.u(4,'NS',q),j]+=1; W[5][n.u(4,'NO',q),j]-=2
        W[6][n.u(5,'acc',p),mv(p//9,p%9)]=1; W[6][n.u(5,'Uns',p),mv(p//9,p%9)]=-prm['uns']
        b[6][mv(p//9,p%9)]=-C
    b[6][81]=prm['pass']
    return n
def default_prm():
    pr=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9); line=min(r,c,8-r,8-c)
        pr[p]=[-1.5,-0.3,0.5,0.6,0.7][line]
    return dict(prior=pr,att=0.5,terr_th=2.0,terr=1.5,eye=12,cap=8,esc=5,uns=5,**{'pass':-1.0})
'''
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
n=build.build(build.default_prm(),inp,mv)
print(n.used)
n.save(save); print(saved())
```