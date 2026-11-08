```python
code = r'''
import numpy as np, sys
sys.path.insert(0,'/work')
from netlib import *
D=lambda a,c: abs(a//9-c//9)+abs(a%9-c%9)
DG=[[q for q in range(81) if abs(q//9-p//9)==1 and abs(q%9-p%9)==1] for p in range(81)]
def build(prm, inp, mv):
    n=Net(); W=n.W; b=n.b
    s_=lambda p: inp(p//9,p%9,0); o_=lambda p: inp(p//9,p%9,1)
    for nm in ['S','O','LS','LO']: n.alloc(0,nm)
    for p in range(81):
        W[0][s_(p),n.u(0,'S',p)]=1; W[0][o_(p),n.u(0,'O',p)]=1
        for col,pl in (('S',s_),('O',o_)):
            j=n.u(0,'L'+col,p)
            for m in NB[p]: W[0][s_(m),j]-=1; W[0][o_(m),j]-=1
            W[0][pl(p),j]+=4; b[0][j]=len(NB[p])-4
    for nm in ['S','O','EvS','EvO','E3S']: n.alloc(1,nm)
    for p in range(81):
        for nm in ['S','O']: W[1][n.u(0,nm,p),n.u(1,nm,p)]=1
        for col in 'SO':
            j=n.u(1,'Ev'+col,p); W[1][n.u(0,'L'+col,p),j]+=2
            for m in NB[p]: W[1][n.u(0,'L'+col,m),j]+=1
            W[1][n.u(0,col,p),j]+=20; b[1][j]=-22
        j=n.u(1,'E3S',p)
        for m in [p]+NB[p]: W[1][n.u(0,'LS',m),j]+=1
        W[1][n.u(0,'S',p),j]+=20; b[1][j]=-22
    for k in (2,3,4):
        for nm in ['S','O','NS','NO','US']: n.alloc(k,nm)
        for p in range(81):
            for nm in ['S','O']: W[k][n.u(k-1,nm,p),n.u(k,nm,p)]=1
            for X,col,Ev in (('N','S','Ev'),('N','O','Ev'),('U','S','E3')):
                j=n.u(k,X+col,p)
                if k==2:
                    W[k][n.u(1,col,p),j]=1
                    for m in [p]+NB[p]: W[k][n.u(1,Ev+col,m),j]-=1
                else:
                    for m in [p]+NB[p]:
                        W[k][n.u(k-1,col,m),j]-=1; W[k][n.u(k-1,X+col,m),j]+=1
                    W[k][n.u(k-1,col,p),j]+=1
    n.alloc(4,'Terr')
    for p in range(81):
        j=n.u(4,'Terr',p); b[4][j]=-prm['terr_th']
        for q in NB[p]: W[4][n.u(3,'S',q),j]+=1; W[4][n.u(3,'O',q),j]-=2; W[4][n.u(3,'NS',q),j]-=10
        for q in range(81):
            if D(p,q)==2: W[4][n.u(3,'S',q),j]+=.5; W[4][n.u(3,'O',q),j]-=1
    for nm in ['Eye','Pos','U2','U3','NS6','NO6']: n.alloc(5,nm)
    u=lambda nm,p: n.u(4,nm,p)
    for p in range(81):
        o=mv(p//9,p%9); d=len(NB[p])
        W[6][u('Terr',p),o]=0  # placeholder (Terr is in L5, needs carry) 
        j=n.u(5,'Eye',p); b[5][j]=-d+1
        for m in NB[p]: W[5][u('S',m),j]+=1
        W[5][u('S',p),j]-=4; W[5][u('O',p),j]-=4
        W[6][j,o]=-prm['eye']
        for col,w in (('S',prm['esc']),('O',prm['cap'])):
            j=n.u(5,'N'+col+'6',p); W[5][u(col,p),j]+=1
            for m in [p]+NB[p]: W[5][u(col,m),j]-=1; W[5][u('N'+col,m),j]+=1
            for q in NB[p]: W[6][j,mv(q//9,q%9)]+=w
        j=n.u(5,'Pos',p); b[5][j]=prm['pos_off']
        W[5][u('Terr',p),j]-=prm['terr']
        for q in NB[p]:
            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']
            W[5][u('O',q),j]+=prm['c_opp']; W[5][u('S',q),j]+=prm['c_own']
        for q in DG[p]: W[5][u('S',q),j]+=prm['c_dg']
        for q in range(81):
            if 1<=D(p,q)<=2: W[5][u('S',q),j]-=prm['c_open']; W[5][u('O',q),j]-=prm['c_open']; b[5][j]+=prm['c_open']
        W[6][j,o]=1; b[6][o]=prm['prior'][p]-prm['pos_off']
        for nm,th,wU,wN in (('U2',1,2,0),('U3',2,0,2)):
            j=n.u(5,nm,p); b[5][j]=th-d
            for q in NB[p]:
                W[5][u('O',q),j]+=1; W[5][u('S',q),j]-=1; W[5][u('US',q),j]+=wU; W[5][u('NS',q),j]+=wN; W[5][u('NO',q),j]-=prm['scap']
            W[6][j,o]=-prm[nm]
    b[6][81]=prm['pass']
    return n
def default_prm():
    pr=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9); line=min(r,c,8-r,8-c)
        pr[p]=[-1.5,-0.5,0,0.2,0.4][line]
    return dict(prior=pr,eye=12,cap=12,esc=12,**{'def':2},c_opp=0.6,c_own=0.4,c_dg=0.3,c_open=0.15,U2=25,U3=25,scap=2,terr=1.5,terr_th=2.0,pos_off=20.0,**{'pass':-4.0})
'''
open('/work/build3.py','w').write(code)
import sys; sys.path.insert(0,'/work'); import build3
n=build3.build(build3.default_prm(),inp,mv); n.save(save); print(n.used)
```