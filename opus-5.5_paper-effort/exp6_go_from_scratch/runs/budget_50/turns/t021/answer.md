```python
code = r'''
import numpy as np, sys
sys.path.insert(0,'/work')
from netlib import *
D=lambda a,c: abs(a//9-c//9)+abs(a%9-c%9)
def build(prm, inp, mv):
    n=Net(); W=n.W; b=n.b; C=100.0
    s_=lambda p: inp(p//9,p%9,0); o_=lambda p: inp(p//9,p%9,1)
    for nm in ['S','O','LS','LS2','LO','LO2']: n.alloc(0,nm)
    for p in range(81):
        W[0][s_(p),n.u(0,'S',p)]=1; W[0][o_(p),n.u(0,'O',p)]=1
        d=len(NB[p])
        for col,pl in (('S',s_),('O',o_)):
            for nm,off in (('L'+col,0),('L'+col+'2',1)):
                j=n.u(0,nm,p)
                for m in NB[p]: W[0][s_(m),j]-=1; W[0][o_(m),j]-=1
                W[0][pl(p),j]+=4; b[0][j]=d-4-off
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
    for k in (2,3,4):
        for nm in ['S','O','NS','NO','acc']: n.alloc(k,nm)
        for p in range(81):
            for nm in ['S','O','acc']: W[k][n.u(k-1,nm,p),n.u(k,nm,p)]=1
            if k==2: W[k][n.u(1,'Eye',p),n.u(k,'acc',p)]=-prm['eye']
            for col in 'SO':
                j=n.u(k,'N'+col,p)
                if k==2:
                    W[k][n.u(1,col,p),j]=1
                    for m in [p]+NB[p]: W[k][n.u(1,'Ev'+col,m),j]-=1
                else:
                    for m in [p]+NB[p]:
                        W[k][n.u(k-1,col,m),j]-=1; W[k][n.u(k-1,'N'+col,m),j]+=1
                    W[k][n.u(k-1,col,p),j]+=1
    for nm in ['acc','Uns','E1','E2']: n.alloc(5,nm)
    K=4
    for p in range(81):
        o=mv(p//9,p%9); j=n.u(5,'acc',p); W[5][n.u(4,'acc',p),j]=1
        for q in NB[p]: W[5][n.u(4,'NO',q),j]+=prm['cap']
        for nm,kk in (('Uns',1),('E1',K),('E2',K)):
            j=n.u(5,nm,p); b[5][j]=kk*(len(NB[p])-2)
            for q in NB[p]:
                W[5][n.u(4,'O',q),j]-=kk; W[5][n.u(4,'NS',q),j]-=kk; W[5][n.u(4,'NO',q),j]+=2*kk
                if nm=='E1': W[5][n.u(4,'NS',q),j]+=1
        # Uns = relu(2-s): bias 2-deg, +O +NS -2NO  -> sign flip
        j=n.u(5,'Uns',p); b[5][j]=2-len(NB[p]); W[5][:,j]*=-1
        W[6][n.u(5,'acc',p),o]=1; W[6][n.u(5,'Uns',p),o]=-prm['uns']
        W[6][n.u(5,'E1',p),o]=prm['esc']; W[6][n.u(5,'E2',p),o]=-prm['esc']
        b[6][o]=-C
    b[6][81]=prm['pass']
    # Voronoi-like gain units Z(m,k), Y(m,k)
    units=[(t,k,m) for k in (1,2,3) for t in 'ZY' for m in range(81) if prm['w'+t][k]!=0]
    L=1
    for t,k,m in units:
        while n.used[L]>=512: L+=1
        j=n.used[L]; n.used[L]+=1; b[L][j]=1
        for q in range(81):
            d=D(q,m)
            if d<=k: W[L][n.u(L-1,'S',q),j]-=1
            if d<=k-(1 if t=='Y' else 0): W[L][n.u(L-1,'O',q),j]-=1
        for p in range(81):
            if D(p,m)==k:
                tgt=n.u(L+1,'acc',p) if L<5 else mv(p//9,p%9)
                (W[L+1] if L<5 else W[6])[j,tgt]+=prm['w'+t][k]
    return n
def default_prm():
    pr=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9); line=min(r,c,8-r,8-c)
        pr[p]=[-1.0,-0.3,0,0,0][line]-3
    return dict(prior=pr,att=0.5,eye=12,cap=8,esc=5,uns=8,wZ=[0,.5,.35,.2],wY=[0,.5,.35,.2],**{'pass':-2.0})
'''
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import build
n=build.build(build.default_prm(),inp,mv)
print(n.used); n.save(save)
```