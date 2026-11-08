
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
    for nm in ['S','O','EvS','EvO','E3S','E3O']: n.alloc(1,nm)
    for p in range(81):
        for nm in ['S','O']: W[1][n.u(0,nm,p),n.u(1,nm,p)]=1
        for col in 'SO':
            j=n.u(1,'Ev'+col,p); W[1][n.u(0,'L'+col,p),j]+=2
            for m in NB[p]: W[1][n.u(0,'L'+col,m),j]+=1
            W[1][n.u(0,col,p),j]+=20; b[1][j]=-22
        for col in 'SO':
            j=n.u(1,'E3'+col,p)
            for m in [p]+NB[p]: W[1][n.u(0,'L'+col,m),j]+=1
            W[1][n.u(0,col,p),j]+=20; b[1][j]=-22
    for k in (2,3,4):
        for nm in (['S','O','NS','NO','US','UO'] if (k<4 or not prm['iso5']) else ['S','O','NS','NO','US','Iso']): n.alloc(k,nm)
        for p in range(81):
            for nm in ['S','O']: W[k][n.u(k-1,nm,p),n.u(k,nm,p)]=1
            if k==4 and prm['iso5']:
                j=n.u(4,'Iso',p); d=len(NB[p]); b[4][j]=4-2*d
                for q in NB[p]: W[4][n.u(3,'NO',q),j]-=1; W[4][n.u(3,'S',q),j]-=3; W[4][n.u(3,'O',q),j]+=2
                W[4][n.u(3,'S',p),j]-=10; W[4][n.u(3,'O',p),j]-=10
            for X,col,Ev in ((('N','S','Ev'),('N','O','Ev'),('U','S','E3'),('U','O','E3')) if (k<4 or not prm['iso5']) else (('N','S','Ev'),('N','O','Ev'),('U','S','E3'))):
                j=n.u(k,X+col,p)
                if k==2:
                    W[k][n.u(1,col,p),j]=1
                    for m in [p]+NB[p]: W[k][n.u(1,Ev+col,m),j]-=1
                else:
                    for m in [p]+NB[p]:
                        W[k][n.u(k-1,col,m),j]-=1; W[k][n.u(k-1,X+col,m),j]+=1
                    W[k][n.u(k-1,col,p),j]+=1
    for nm in ['Eye','Pos','U2','U3','NS6','Terr']: n.alloc(5,nm)
    u=lambda nm,p: n.u(4,nm,p)
    for p in range(81):
        o=mv(p//9,p%9); d=len(NB[p])
        j=n.u(5,'Eye',p); b[5][j]=-d+1
        for m in NB[p]: W[5][u('S',m),j]+=1
        W[5][u('S',p),j]-=4; W[5][u('O',p),j]-=4
        W[6][j,o]=-prm['eye']
        j=n.u(5,'Terr',p); b[5][j]=-prm['terr_th']
        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2; W[5][u('NS',q),j]-=10
        for q in range(81):
            if D(p,q)==2: W[5][u('S',q),j]+=.5; W[5][u('O',q),j]-=1
        W[6][j,o]=-prm['terr']
        if prm['iso6']:
            j=n.u(5,'NS6',p); b[5][j]=4-2*d
            for q in NB[p]: W[5][u('NO',q),j]-=1; W[5][u('S',q),j]-=3; W[5][u('O',q),j]+=2
            W[5][u('S',p),j]-=10; W[5][u('O',p),j]-=10
            W[6][j,o]=-prm['iso']
        for col,w in ((('S',prm['esc']),) if not prm['iso6'] else ()):
            j=n.u(5,'N'+col+'6',p); W[5][u(col,p),j]+=1
            for m in [p]+NB[p]: W[5][u(col,m),j]-=1; W[5][u('N'+col,m),j]+=1
            for q in NB[p]: W[6][j,mv(q//9,q%9)]+=w
        j=n.u(5,'Pos',p); b[5][j]=prm['pos_off']
        for q in NB[p]:
            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']
            if prm['iso6']: W[5][u('NS',q),j]+=prm['esc']
            if prm['iso5']: W[5][u('NO',q),j]+=prm['cap']
            else: W[5][u('UO',q),j]+=prm['atk']; W[5][u('NO',q),j]+=prm['cap']-prm['atk']
            W[5][u('O',q),j]+=prm['c_opp']; W[5][u('S',q),j]+=prm['c_own']
        if prm['iso5']: W[5][u('Iso',p),j]-=prm['iso']
        for q in DG[p]: W[5][u('S',q),j]+=prm['c_dg']
        for q in range(81):
            if 1<=D(p,q)<=2: W[5][u('S',q),j]-=prm['c_open']; W[5][u('O',q),j]-=prm['c_open']; b[5][j]+=prm['c_open']
        W[6][j,o]=1; b[6][o]=prm['prior'][p]-prm['pos_off']
        for nm,c,al,be,sc in (('U2',1,.9,.3,prm['scap']),('U3',2.5,1,.5,prm['scap3'])):
            j=n.u(5,nm,p); b[5][j]=c-al*d+(prm['kap']*(2-d) if nm=='U3' else 0)
            for q in NB[p]:
                if nm=='U3': W[5][u('O',q),j]+=prm['kap']
                W[5][u('O',q),j]+=al; W[5][u('S',q),j]-=al; W[5][u('US',q),j]+=2*al-be; W[5][u('NS',q),j]+=be; W[5][u('NO',q),j]-=al*sc
            W[6][j,o]=-prm[nm]
    b[6][81]=prm['pass']
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
    # phase: stones on board, clamped at 50 -> per-point prior shift
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
    W[6][n.blk[5]['Score'],81]=prm['pw']
    W[6][n.blk[5]['Ph1'],81]=prm['pph']; W[6][n.blk[5]['Ph2'],81]=-prm['pph']
    b[6][81]=prm['pass']-100*prm['pw']-50*prm['pph']
    for nm,th,w in (('Sc25',125,-prm['pw']),('Sc10',100+prm['pth1'],prm['pw2']),('Sc25b',125,-prm['pw2'])):
        j=n.alloc(5,nm,1); W[5][n.blk[4]['Score'],j]=1; b[5][j]=-th; W[6][j,81]=w
    return n
def default_prm():
    pr=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9); line=min(r,c,8-r,8-c)
        pr[p]=[0,0,0,0.2,0.4][line]
    return dict(prior=pr,eye=12,cap=12,esc=12,**{'def':4},c_opp=0.6,c_own=0.4,c_dg=0.3,c_open=0.15,U2=250,U3=6,atk=4,scap=2,scap3=0,terr=1.5,terr_th=2.0,pos_off=20.0,komi=7.5,iso6=False,iso5=False,kap=0.0,te=2.5,te_deg=1.0,iso=6,pth1=10,pw2=0.4,pph=0.3,early=[-3,-1.2,0,0,0],late=[-0.5,0,0,0,0],tw=0.6,pw=0.4,**{'pass':-4.0})
