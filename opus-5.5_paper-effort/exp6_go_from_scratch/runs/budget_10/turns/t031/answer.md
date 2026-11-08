```python
code = open('/work/build.py').read()
# 1) phase unit and tengen prior; scale tau
code = code.replace("""    W6[0:162,486]=1
    W6[486,487]=1
    Wo[486,81]=P['c_nst']; bo[81]=P['b_pass']""","""    W6[0:162,486]=1
    W6[486,487]=1
    Wo[486,81]=P['c_nst']; bo[81]=P['b_pass']
    # phase unit: relu(T - #stones)
    W6[0:162,488]=-1; b6[488]=P['phT']
    for i in range(81):
        r,c=divmod(i,9); d=min(r,c,8-r,8-c)
        Wo[488,i]=-P['tau']*P['phase_pen'].get(d,0.0)/P['phT']
    bo[40]+=P['tau']*P['tengen']""")
code = code.replace("""        Wo[u,i]=1.0; bo[i]-=off""","""        Wo[u,i]=P['tau']; bo[i]-=off*P['tau']""")
code = code.replace("""        bo[i]+=const
""","""        bo[i]+=const*P['tau']
""")
code = code.replace("""                Wo[324+i,p]-=P['gamma']*w
                Wo[405+i,p]-=P['delta']*w""","""                Wo[324+i,p]-=P['tau']*P['gamma']*w
                Wo[405+i,p]-=P['tau']*P['delta']*w""")
code = code.replace("""        bo[i]+=P['prior'][d]""","""        bo[i]+=P['tau']*P['prior'][d]""")
# new defaults
s=code.index("DEF=dict("); e=code.index("def build(")
newdef='''DEF=dict(C=10000.0,
    a1=10.0,a2=3.0,a3=7.0,a4=2.0,
    w_eye=30.0, w_sa=10.0,
    J=dict(o1=-0.2,x1=0.4,od=0.3,xd=0.2,o2=0.2,x2=0.1,ok=0.15,xk=0.05),
    IK={(0,1):1.0,(1,1):0.7,(0,2):0.5,(1,2):0.35,(2,2):0.25,(0,3):0.2},
    VK={(0,0):1.0,(0,1):0.8,(1,1):0.6,(0,2):0.45,(1,2):0.3,(2,2):0.15,(0,3):0.15},
    ia=0.5, beta=0.25, gamma=0.5, delta=0.35, tau=2.0,
    prior={0:-1.0,1:-0.3,2:0.0,3:0.0,4:0.0}, tengen=0.4,
    phT=40.0, phase_pen={0:2.0,1:1.0},
    b_pass=-12.0, c_nst=0.12)

'''
code=code[:s]+newdef+code[e:]
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
print(W['out.b'].reshape(-1)[:81].reshape(9,9).round(2)); print(W['out.b'][81])
print(W['out.W'][488,:9])
```