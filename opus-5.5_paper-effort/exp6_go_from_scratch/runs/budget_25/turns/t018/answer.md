
<invoke name="mcp__sandbox__python">
<parameter name="code">print(1)</parameter>
</invoke>

<function_results>
<system>Tool ran without output or errors</system>
</function_results>

<br>

```python
code = r'''
import sys; sys.path.insert(0,'/work')
import numpy as np
import netlib_ab as NL
NB,EDGES,IO,IP=NL.NB,NL.EDGES,NL.IO,NL.IP
def par(s): return (s//9+s%9)%2
def p3(s):
    r,c=divmod(s,9)
    if (r+c)%2==0: return 0
    return 1 if r%2==0 else 2
def line(s):
    r,c=divmod(s,9); return min(r,c,8-r,8-c)+1
def own_colors(q):
    return [1,2] if par(q)==1 else [0,1,2]
def build(P):
    net=NL.Net(); U={}
    M=1000.0
    # ---- L1
    for s in range(81):
        U[('b',1,s)]=net.unit(1,('b',s),[(IO(s),1),(IP(s),-1)],1.0)
    for s in range(81):
        deg=len(NB[s])
        Eterms=lambda ns,w: [(IO(n),-w) for n in ns]+[(IP(n),-w) for n in ns]
        # own P3 seeds; m = sum_n (1-O_n-P_n) = deg - sum O - sum P
        if par(s)==1:
            for k,th in ((0,5),(1,6),(2,7)):
                U[('o',1,s,k)]=net.unit(1,('o',s,k),[(IO(s),5)]+Eterms(NB[s],1),deg-th)
        else:
            nBs=[n for n in NB[s] if p3(n)==1]; nCs=[n for n in NB[s] if p3(n)==2]
            U[('o',1,s,1)]=net.unit(1,('o',s,1),[(IO(s),5)]+Eterms(nBs,2)+Eterms(nCs,1),2*len(nBs)+len(nCs)-6)
            U[('o',1,s,2)]=net.unit(1,('o',s,2),[(IO(s),5)]+Eterms(nCs,2)+Eterms(nBs,1),2*len(nCs)+len(nBs)-6)
            U[('o',1,s,0)]=net.unit(1,('o',s,0),[(IO(s),5)]+Eterms(NB[s],1),deg-7)
        # opp par2 seeds
        U[('x',1,s,1-par(s))]=net.unit(1,('x',s,1-par(s)),[(IP(s),5)]+Eterms(NB[s],1),deg-5)
        U[('x',1,s,par(s))]=net.unit(1,('x',s,par(s)),[(IP(s),5)]+Eterms(NB[s],1),deg-6)
    # ---- L2..L4 propagation
    for l in (2,3,4):
        for s in range(81):
            U[('b',l,s)]=net.unit(l,('b',s),[(U[('b',l-1,s)],1.0)],0.0)
        for s in range(81):
            for k in (0,1,2):
                terms=[(U[('o',l-1,t,k)],1.0) for t in [s]+NB[s] if ('o',l-1,t,k) in U]
                U[('o',l,s,k)]=net.unit(l,('o',s,k),terms+[(U[('b',l-1,s)],M)],-2*M)
            for k in (0,1):
                terms=[(U[('x',l-1,t,k)],1.0) for t in [s]+NB[s] if ('x',l-1,t,k) in U]
                U[('x',l,s,k)]=net.unit(l,('x',s,k),terms+[(U[('b',l-1,s)],-M)],0.0)
    # ---- L5
    for s in range(81):
        U[('b',5,s)]=net.unit(5,('b',s),[(U[('b',4,s)],1.0)],0.0)
        U[('O',5,s)]=net.unit(5,('O',s),[(U[('b',4,s)],2.0)],-3.0)
    for q in range(81):
        for k in own_colors(q):
            terms=[(U[('o',4,t,k)],-1.0) for t in [q]+NB[q]]
            U[('m',5,q,k)]=net.unit(5,('m',q,k),terms+[(U[('b',4,q)],1.0)],-1.0)
        k=par(q)
        terms=[(U[('x',4,t,k)],-1.0) for t in [q]+NB[q]]
        U[('n',5,q)]=net.unit(5,('n',q),terms+[(U[('b',4,q)],-2.0)],1.0)
    # ---- L6
    eps=P['eps']
    for p in range(81):
        terms=[(U[('b',5,p)],1.0)]+[(U[('n',5,q)],eps*P['CAP']) for q in NB[p]]
        U[('b',6,p)]=net.unit(6,('b',p),terms,0.0)
        U[('O',6,p)]=net.unit(6,('O',p),[(U[('O',5,p)],1.0)],0.0)
    for q in range(81):
        U[('at',6,q)]=net.unit(6,('at',q),[(U[('m',5,q,k)],1.0) for k in own_colors(q)],-1.0)
    for p in range(81):
        deg=len(NB[p])
        ne=[(U[('b',5,e)],1.0) for e in NB[p]]+[(U[('O',5,e)],-2.0) for e in NB[p]]
        # U1 = relu(B1*(1.5-ne) - BIG1*sum_q (2O_q - sum_k m_qk))
        t1=[(i,-P['B1']*w) for i,w in ne]
        for q in NB[p]:
            t1.append((U[('O',5,q)],-P['BIG1']*2.0))
            for k in own_colors(q): t1.append((U[('m',5,q,k)],P['BIG1']))
        U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*1.5)
        t0=[(i,-P['B0']*w) for i,w in ne]
        for q in NB[p]:
            t0.append((U[('O',5,q)],-P['BIG0']*1.0))
            for k in own_colors(q): t0.append((U[('m',5,q,k)],P['BIG0']))
        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*0.5)
        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))
    # ---- output
    for p in range(81):
        r,c=divmod(p,9)
        terms=[(U[('b',6,p)],1.0/eps)]
        terms+=[(U[('at',6,q)],P['ESC']) for q in NB[p]]
        terms+=[(U[('U1',6,p)],-P['SA1']),(U[('U0',6,p)],-P['SA0']),(U[('eye',6,p)],-P['EYE'])]
        bias=-1.0/eps+P['LV'][line(p)]
        # linear proximity: own O6, opp P = 1+O-b
        for t in range(81):
            dd=abs(t//9-r)+abs(t%9-c)
            if dd==0 or dd>2: continue
            wo=P['prox_own'][dd]; wp=P['prox_opp'][dd]
            # P_t = 1 + O_t - b_t  (b6 has eps-perturbation, negligible)
            terms+=[(U[('O',6,t)],wo+wp),(U[('b',6,t)],-wp)]
            bias+=wp
        net.outw(p,terms,bias)
    net.outw(81,[],P['PASS'])
    return net,U
'''
open('/work/v2_ab.py','w').write(code)
import importlib, sys; sys.path.insert(0,'/work')
import v2_ab; importlib.reload(v2_ab)
P=dict(eps=1e-4,CAP=15.0,ESC=12.0,B1=8.0,BIG1=20.0,B0=8.0,BIG0=10.0,SA1=4.0,SA0=3.0,EYE=40.0,
       LV={1:-2.0,2:0.0,3:1.5,4:1.5,5:1.0},prox_own={1:0.3,2:0.15},prox_opp={1:0.5,2:0.3},PASS=-12.0)
net,U=v2_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v2_units.pkl','wb'))
```