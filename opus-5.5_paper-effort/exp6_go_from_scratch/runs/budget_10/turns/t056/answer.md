```python
code = open('/work/build.py').read()
s=code.index("    # ---- L6 features")
e=code.index("    W.update(")
new_l6 = '''    # ---- L6 features
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
'''
code=code[:s]+new_l6+code[e:]
code=code.replace("ia=0.5, beta=0.25, gamma=0.25, delta=0.2, tau=2.0,","ia=0.5, beta=0.25, gamma=0.25, alpha=0.15, tau=2.0, w_cc=3.0,")
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
print({k:v.shape for k,v in W.items()})
print(W['out.b'][:81].reshape(9,9).round(1)); print(W['out.b'][81])
print('J offsets min', W['L6.b'][243:324].min(), W['L6.b'][243:324].max())
```