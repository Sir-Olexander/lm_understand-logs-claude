
import numpy as np, sys
sys.path.insert(0,'/work')
from go import NB, DIAG

def ring(i, dists):
    r,c=divmod(i,9); out=[]
    for rr in range(9):
        for cc in range(9):
            d=abs(rr-r)+abs(cc-c)
            if d in dists: out.append(rr*9+cc)
    return out

def knight(i):
    r,c=divmod(i,9); out=[]
    for dr,dc in ((1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out
def straight2(i):
    r,c=divmod(i,9); out=[]
    for dr,dc in ((2,0),(-2,0),(0,2),(0,-2)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out

def kern(i,j,K):
    r1,c1=divmod(i,9); r2,c2=divmod(j,9)
    dr,dc=abs(r1-r2),abs(c1-c2)
    key=(min(dr,dc),max(dr,dc))
    return K.get(key,0.0)

DEF=dict(C=10000.0,
    a1=9.0,a2=3.0,a3=6.0,a4=2.0,
    mu_opp=0.0, d0_opp=0.0, mu_own=0.0, d0_own=0.0, Kk=3.0,
    w_eye=30.0, w_sa=15.0, sa_kill=2.0,
    J=dict(o1=-0.2,x1=0.4,od=0.3,xd=0.2,o2=0.2,x2=0.1,ok=0.15,xk=0.05),
    IK={(0,1):1.0,(1,1):0.7,(0,2):0.5,(1,2):0.35,(2,2):0.25,(0,3):0.2},
    VK={(0,0):1.0,(0,1):0.8,(1,1):0.6,(0,2):0.45,(1,2):0.3,(2,2):0.15,(0,3):0.15},
    PK={(0,0):1.0,(0,1):0.5},
    ia=0.5, beta=0.25, gamma=0.25, alpha=0.15, tau=2.0, w_cc=4.0,
    prior={0:-1.0,1:-0.3,2:0.0,3:0.0,4:0.0}, tengen=0.4, seed='mix',
    phT=40.0, phase_pen={0:2.0,1:1.0},
    b_pass=-12.0, c_nst=0.13)

def build(P=DEF):
    W={}
    C=P['C']
    # ---- L1
    W1=np.zeros((243,512),np.float32); b1=np.zeros(512,np.float32)
    for i in range(81):
        W1[i*3+0,i]=1; W1[i*3+1,81+i]=1
        deg=len(NB[i])
        for base,pl in ((162,0),(243,1)):
            u=base+i
            for j in NB[i]:
                W1[j*3+0,u]-=1; W1[j*3+1,u]-=1
            W1[i*3+pl,u]+=4
            b1[u]=deg-4
    W1[2,324]=1
    # ---- L2
    W2=np.zeros((512,512),np.float32); b2=np.zeros(512,np.float32)
    for k in range(162): W2[k,k]=1
    for i in range(81):
        for (Hb,Sb,Fb,F3b) in ((162,0,162,324),(243,81,243,405)):
            if P.get('seed','mix')=='old':
                u=Fb+i
                W2[Hb+i,u]+=16
                for j in NB[i]: W2[Hb+j,u]+=1
                W2[Sb+i,u]+=20; b2[u]=-36
                u=F3b+i
                W2[Hb+i,u]+=1
                for j in NB[i]: W2[Hb+j,u]+=1
                W2[Sb+i,u]+=20; b2[u]=-22
            else:
                # exact F seeds: libs >= e + ceil(sum_nbr_e / 2)
                u=Fb+i
                W2[Hb+i,u]+=5
                for j in NB[i]: W2[Hb+j,u]+=2
                W2[Sb+i,u]+=40; b2[u]=-45
                u=F3b+i
                if P.get('seed','mix')=='exact':
                    W2[Hb+i,u]+=4
                    for j in NB[i]: W2[Hb+j,u]+=2
                    W2[Sb+i,u]+=40; b2[u]=-49
                else:   # mix: 1.5e + sum >= 4
                    W2[Hb+i,u]+=3
                    for j in NB[i]: W2[Hb+j,u]+=2
                    W2[Sb+i,u]+=40; b2[u]=-47
    W2[324,486]=1
    # ---- L3, L4 propagation
    def prop():
        Wp=np.zeros((512,512),np.float32); bp=np.zeros(512,np.float32)
        for k in range(162): Wp[k,k]=1
        for i in range(81):
            for (B,Sb) in ((162,0),(243,81),(324,0),(405,81)):
                u=B+i
                for j in [i]+NB[i]: Wp[B+j,u]+=1
                Wp[Sb+i,u]+=C; bp[u]=-C
        Wp[486,486]=1
        return Wp,bp
    W3,b3=prop(); W4,b4=prop()
    # ---- L5 saturation
    W5=np.zeros((512,512),np.float32); b5=np.zeros(512,np.float32)
    for k in range(162): W5[k,k]=1
    for i in range(81):
        cl=[i]+NB[i]
        # A_own / A_opp graded by same-colour density within distance 2
        R2=ring(i,(1,2))
        u=162+i; W5[i,u]=1+P['mu_own']*P['d0_own']
        for j in cl: W5[162+j,u]-=P['Kk']; W5[324+j,u]-=P['Kk']
        for j in R2: W5[j,u]-=P['mu_own']
        u=243+i; W5[81+i,u]=1+P['mu_opp']*P['d0_opp']
        for j in cl: W5[243+j,u]-=P['Kk']; W5[405+j,u]-=P['Kk']
        for j in R2: W5[81+j,u]-=P['mu_opp']
        u=324+i; W5[i,u]=1
        for j in cl: W5[324+j,u]-=1
        u=405+i; W5[81+i,u]=1
        for j in cl: W5[405+j,u]-=1
    W5[486,486]=1
    # ---- L6 features
    W6=np.zeros((512,512),np.float32); b6=np.zeros(512,np.float32)
    Wo=np.zeros((512,82),np.float32); bo=np.zeros(82,np.float32)
    Jw=P['J']; tau=P['tau']
    NSTu, PHu = 509, 510
    for i in range(81):
        deg=len(NB[i])
        # U : stone status -> neighbours
        u=i
        W6[243+i,u]=P['a1']; W6[405+i,u]=P['a2']; W6[162+i,u]=P['a3']; W6[324+i,u]=P['a4']
        for j in NB[i]: Wo[u,j]=1.0
        # EYE
        u=81+i
        for j in NB[i]: W6[j,u]+=1; W6[162+j,u]-=1
        W6[i,u]-=4; W6[81+i,u]-=4; b6[u]=1-deg
        Wo[u,i]=-P['w_eye']
        # SA
        u=162+i
        b6[u]=2-deg
        for j in NB[i]:
            W6[j,u]-=1; W6[81+j,u]+=1; W6[324+j,u]+=2; W6[243+j,u]-=P['sa_kill']
        Wo[u,i]=-P['w_sa']
        # J : linear terms (offset keeps it positive)
        u=243+i
        lin=np.zeros(162)
        def addJ(pts,wo,wx):
            for j in pts: lin[j]+=wo; lin[81+j]+=wx
        addJ(NB[i],Jw['o1'],Jw['x1']); addJ(DIAG[i],Jw['od'],Jw['xd'])
        addJ(straight2(i),Jw['o2'],Jw['x2']); addJ(knight(i),Jw['ok'],Jw['xk'])
        const=0.0
        for j in range(81):
            w=kern(i,j,P['VK'])
            if w:
                const+=P['beta']*w; lin[j]-=P['beta']*w; lin[81+j]-=P['beta']*w
        # linear influence term alpha * sum_{x in N[p]} PK * I(x)
        for x in range(81):
            w=kern(i,x,P['PK'])
            if not w: continue
            for j in range(81):
                k=kern(x,j,P['IK'])
                if k and j!=x:
                    lin[j]+=P['alpha']*w*k; lin[81+j]-=P['alpha']*w*k
        off=-np.minimum(lin[:81],lin[81:]).clip(max=0).sum()+1.0
        W6[0:162,u]=lin; b6[u]=off
        Wo[u,i]=tau; bo[i]+=tau*(const-off)
        # POS (masked to empty points)
        u=324+i
        for j in range(81):
            k=kern(i,j,P['IK'])
            if k and j!=i:
                W6[j,u]+=k; W6[81+j,u]-=k
        W6[i,u]-=50; W6[81+i,u]-=50
        b6[u]=-P['ia']
        for p in range(81):
            w=kern(p,i,P['PK'])
            if w: Wo[u,p]-=tau*(P['gamma']+P['alpha'])*w
        # static prior
        r,c=divmod(i,9); d=min(r,c,8-r,8-c)
        bo[i]+=tau*P['prior'][d]
    if P.get('neg_mode',False):
        # NEG units at 405..485 (pass-5 style), no CC
        for i in range(81):
            u=405+i
            for j in range(81):
                k=kern(i,j,P['IK'])
                if k and j!=i:
                    W6[j,u]-=k; W6[81+j,u]+=k
            W6[i,u]-=50; W6[81+i,u]-=50
            b6[u]=-P['ia']
            for p in range(81):
                w=kern(p,i,P['PK'])
                if w: Wo[u,p]-=tau*P['delta']*w
        u=486
    else:
        # CC: cut / connect patterns on 2x2 blocks
        skip=set([(0,0),(0,7),(7,0),(7,7),(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)])
        u=405
        for r in range(8):
            for c in range(8):
                if (r,c) in skip: continue
                a=r*9+c; bq=r*9+c+1; d_=(r+1)*9+c; e_=(r+1)*9+c+1
                for (o1,o2,x1,x2) in ((a,e_,bq,d_),(bq,d_,a,e_)):
                    W6[o1,u]+=1; W6[o2,u]+=1; W6[81+x1,u]+=1; W6[81+x2,u]+=1
                    b6[u]=-2.5
                    for t in (a,bq,d_,e_): Wo[u,t]+=2*P['w_cc']
                    u+=1
    assert u<=NSTu, u
    W6[0:162,NSTu]=1
    Wo[NSTu,81]=P['c_nst']; bo[81]=P['b_pass']
    W6[0:162,PHu]=-1; b6[PHu]=P['phT']
    for i in range(81):
        r,c=divmod(i,9); d=min(r,c,8-r,8-c)
        Wo[PHu,i]=-tau*P['phase_pen'].get(d,0.0)/P['phT']
    bo[40]+=tau*P['tengen']
    W.update({'L1.W':W1,'L1.b':b1,'L2.W':W2,'L2.b':b2,'L3.W':W3,'L3.b':b3,'L4.W':W4,'L4.b':b4,
              'L5.W':W5,'L5.b':b5,'L6.W':W6,'L6.b':b6,'out.W':Wo,'out.b':bo})
    return W
