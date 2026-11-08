```python
code = r'''
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

DEF=dict(C=10000.0,
    a1=7.5,a2=2.5,a3=4.5,a4=1.5,
    w_eye=25.0, w_sa=7.0,
    J=dict(o1=0.2,x1=0.7,od=0.5,xd=0.2,o2=0.4,x2=0.2,ok=0.2,xk=0.1),
    ot=dict(beta=0.1,theta=4.0,xw=5.0,w=1.0),
    xt=dict(beta=0.05,theta=3.0,ow=5.0,w=0.5),
    prior={0:-1.5,1:-0.4,2:0.4,3:0.3,4:0.6},
    b_pass=-9.0, c_nst=0.08)

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
            u=Fb+i
            W2[Hb+i,u]+=16
            for j in NB[i]: W2[Hb+j,u]+=1
            W2[Sb+i,u]+=20; b2[u]=-36
            u=F3b+i
            W2[Hb+i,u]+=1
            for j in NB[i]: W2[Hb+j,u]+=1
            W2[Sb+i,u]+=20; b2[u]=-22
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
        # A_own
        u=162+i; W5[i,u]=1
        for j in cl: W5[162+j,u]-=1; W5[324+j,u]-=1
        u=243+i; W5[81+i,u]=1
        for j in cl: W5[243+j,u]-=1; W5[405+j,u]-=1
        u=324+i; W5[i,u]=1
        for j in cl: W5[324+j,u]-=1
        u=405+i; W5[81+i,u]=1
        for j in cl: W5[405+j,u]-=1
    W5[486,486]=1
    # ---- L6 features
    W6=np.zeros((512,512),np.float32); b6=np.zeros(512,np.float32)
    Wo=np.zeros((512,82),np.float32); bo=np.zeros(82,np.float32)
    Jw=P['J']; ot=P['ot']; xt=P['xt']
    for i in range(81):
        deg=len(NB[i])
        # U
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
            W6[j,u]-=1; W6[81+j,u]+=1; W6[324+j,u]+=2; W6[243+j,u]-=3
        Wo[u,i]=-P['w_sa']
        # J
        u=243+i
        neg=0.0
        def addJ(pts,wo,wx):
            nonlocal neg
            for j in pts:
                W6[j,u]+=wo; W6[81+j,u]+=wx
                neg+=min(wo,0)+min(wx,0)
        addJ(NB[i],Jw['o1'],Jw['x1']); addJ(DIAG[i],Jw['od'],Jw['xd'])
        addJ(straight2(i),Jw['o2'],Jw['x2']); addJ(knight(i),Jw['ok'],Jw['xk'])
        off=-neg
        b6[u]=off
        Wo[u,i]=1.0; bo[i]-=off
        # OT
        u=324+i
        for j in ring(i,(1,2)):
            W6[j,u]+=1; W6[81+j,u]-=ot['xw']
        W6[0:162,u]+=ot['beta']
        b6[u]=-ot['theta']
        Wo[u,i]=-ot['w']
        # XT
        u=405+i
        for j in ring(i,(1,2)):
            W6[81+j,u]+=1; W6[j,u]-=xt['ow']
        W6[0:162,u]+=xt['beta']
        b6[u]=-xt['theta']
        Wo[u,i]=-xt['w']
        # prior
        r,c=divmod(i,9); d=min(r,c,8-r,8-c)
        bo[i]+=P['prior'][d]
    W6[0:162,486]=1
    W6[486,487]=1
    Wo[486,81]=P['c_nst']; bo[81]=P['b_pass']
    W.update({'L1.W':W1,'L1.b':b1,'L2.W':W2,'L2.b':b2,'L3.W':W3,'L3.b':b3,'L4.W':W4,'L4.b':b4,
              'L5.W':W5,'L5.b':b5,'L6.W':W6,'L6.b':b6,'out.W':Wo,'out.b':bo})
    return W
'''
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items():
    assert v.shape==SHAPES[k], (k,v.shape)
    save(k,v.astype(np.float32) if hasattr(v,'astype') else v)
print(saved())
```