```python
code = open('/work/build.py').read()
# graded A in L5
code = code.replace("""        # A_own
        u=162+i; W5[i,u]=1
        for j in cl: W5[162+j,u]-=1; W5[324+j,u]-=1
        u=243+i; W5[81+i,u]=1
        for j in cl: W5[243+j,u]-=1; W5[405+j,u]-=1""","""        # A_own / A_opp graded by same-colour density within distance 2
        R2=ring(i,(1,2))
        u=162+i; W5[i,u]=1+P['mu_own']*P['d0_own']
        for j in cl: W5[162+j,u]-=P['Kk']; W5[324+j,u]-=P['Kk']
        for j in R2: W5[j,u]-=P['mu_own']
        u=243+i; W5[81+i,u]=1+P['mu_opp']*P['d0_opp']
        for j in cl: W5[243+j,u]-=P['Kk']; W5[405+j,u]-=P['Kk']
        for j in R2: W5[81+j,u]-=P['mu_opp']""")
code = code.replace("""        for p in range(81):
            w=kern(p,i,P['VK'])
            if w:
                Wo[324+i,p]-=P['tau']*P['gamma']*w
                Wo[405+i,p]-=P['tau']*P['delta']*w""","""        for p in range(81):
            w=kern(p,i,P['PK'])
            if w:
                Wo[324+i,p]-=P['tau']*P['gamma']*w
                Wo[405+i,p]-=P['tau']*P['delta']*w""")
s=code.index("DEF=dict("); e=code.index("def build(")
newdef='''DEF=dict(C=10000.0,
    a1=6.0,a2=3.0,a3=4.0,a4=2.0,
    mu_opp=1/3., d0_opp=3.0, mu_own=0.2, d0_own=5.0, Kk=3.0,
    w_eye=30.0, w_sa=15.0,
    J=dict(o1=-0.2,x1=0.4,od=0.3,xd=0.2,o2=0.2,x2=0.1,ok=0.15,xk=0.05),
    IK={(0,1):1.0,(1,1):0.7,(0,2):0.5,(1,2):0.35,(2,2):0.25,(0,3):0.2},
    VK={(0,0):1.0,(0,1):0.8,(1,1):0.6,(0,2):0.45,(1,2):0.3,(2,2):0.15,(0,3):0.15},
    PK={(0,0):1.0,(0,1):0.5},
    ia=0.5, beta=0.25, gamma=0.25, delta=0.2, tau=2.0,
    prior={0:-1.0,1:-0.3,2:0.0,3:0.0,4:0.0}, tengen=0.4,
    phT=40.0, phase_pen={0:2.0,1:1.0},
    b_pass=-12.0, c_nst=0.13)

'''
code=code[:s]+newdef+code[e:]
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
print(W['L5.W'][:,162+40].nonzero()[0][:40], W['L5.W'][40,162+40], W['L5.W'][41,162+40])
print(W['L5.W'][81+40,243+40], W['L5.W'][81+42,243+40], W['L5.W'][243+40,243+40])
```