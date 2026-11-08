```python
code = open('/work/build.py').read()
code = code.replace("""DEF=dict(C=10000.0,""","""def kern(i,j,K):
    r1,c1=divmod(i,9); r2,c2=divmod(j,9)
    dr,dc=abs(r1-r2),abs(c1-c2)
    key=(min(dr,dc),max(dr,dc))
    return K.get(key,0.0)

DEF=dict(C=10000.0,""")
# replace OT/XT section with POS/NEG influence
old_start=code.index("        # OT")
old_end=code.index("        # prior")
new_sec='''        # POS / NEG influence (masked to empty points)
        IK=P['IK']
        for (u,sgn) in ((324+i,1.0),(405+i,-1.0)):
            for j in range(81):
                k=kern(i,j,IK)
                if k and j!=i:
                    W6[j,u]+=sgn*k; W6[81+j,u]-=sgn*k
            W6[i,u]-=50; W6[81+i,u]-=50
            b6[u]=-P['ia']
        for p in range(81):
            w=kern(p,i,P['VK'])
            if w:
                Wo[324+i,p]-=P['gamma']*w
                Wo[405+i,p]-=P['delta']*w
'''
code=code[:old_start]+new_sec+code[old_end:]
# add empty-kernel term into J: beta * sum_x w(p,x) * (1 - own(x) - opp(x))
code=code.replace("""        addJ(straight2(i),Jw['o2'],Jw['x2']); addJ(knight(i),Jw['ok'],Jw['xk'])
""","""        addJ(straight2(i),Jw['o2'],Jw['x2']); addJ(knight(i),Jw['ok'],Jw['xk'])
        const=0.0
        for j in range(81):
            w=kern(i,j,P['VK'])
            if w:
                const+=P['beta']*w
                W6[j,u]-=P['beta']*w; W6[81+j,u]-=P['beta']*w
                neg-=P['beta']*w
        bo[i]+=const
""")
code=code.replace("""    ot=dict(beta=0.1,theta=4.0,xw=5.0,w=1.0),
    xt=dict(beta=0.05,theta=3.0,ow=5.0,w=0.5),
    prior={0:-1.5,1:-0.4,2:0.4,3:0.3,4:0.6},""","""    IK={(0,1):1.0,(1,1):0.7,(0,2):0.5,(1,2):0.35,(2,2):0.25,(0,3):0.2},
    VK={(0,0):1.0,(0,1):0.8,(1,1):0.6,(0,2):0.45,(1,2):0.3,(2,2):0.15,(0,3):0.15},
    ia=0.5, beta=0.3, gamma=0.3, delta=0.2,
    prior={0:-1.0,1:-0.2,2:0.0,3:0.0,4:0.0},""")
code=code.replace("J=dict(o1=0.2,x1=0.7,od=0.5,xd=0.2,o2=0.4,x2=0.2,ok=0.2,xk=0.1)","J=dict(o1=0.0,x1=0.6,od=0.3,xd=0.3,o2=0.2,x2=0.2,ok=0.1,xk=0.1)")
code=code.replace("Jw=P['J']; ot=P['ot']; xt=P['xt']","Jw=P['J']")
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
print(W['out.b'][:9], W['out.b'][36:45], W['out.b'][81])
# check J pre nonnegativity: offset vs min
W6=W['L6.W']; b6=W['L6.b']
mins=[b6[243+i]+np.minimum(W6[:162,243+i].reshape(2,81),0).sum(axis=0).max()*0 + np.minimum(W6[:81,243+i],W6[81:162,243+i]).clip(max=0).sum() for i in range(81)]
print(min(mins))
```