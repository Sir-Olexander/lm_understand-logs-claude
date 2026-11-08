
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
SKIPP=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44])
BSUB=[s for s in range(81) if s not in SKIPP]
def build(P):
    net=NL.Net(); U={}
    eps=P['eps']
    Ml={2:25.0,3:125.0,4:625.0}
    # ---- L1
    for s in range(81):
        deg=len(NB[s])
        terms=[(IO(s),1-5*eps),(IP(s),-1-5*eps)]+[(IO(n),-eps) for n in NB[s]]+[(IP(n),-eps) for n in NB[s]]
        U[('b',1,s)]=net.unit(1,('b',s),terms,1.0+eps*(1+deg))
    for s in range(81):
        deg=len(NB[s])
        Eterms=lambda ns,w: [(IO(n),-w) for n in ns]+[(IP(n),-w) for n in ns]
        if par(s)==1:
            for k,th in ((0,5),(1,6),(2,7)):
                U[('o',1,s,k)]=net.unit(1,('o',s,k),[(IO(s),5)]+Eterms(NB[s],1),deg-th)
        else:
            nBs=[n for n in NB[s] if p3(n)==1]; nCs=[n for n in NB[s] if p3(n)==2]
            U[('o',1,s,1)]=net.unit(1,('o',s,1),[(IO(s),5)]+Eterms(nBs,2)+Eterms(nCs,1),2*len(nBs)+len(nCs)-6)
            U[('o',1,s,2)]=net.unit(1,('o',s,2),[(IO(s),5)]+Eterms(nCs,2)+Eterms(nBs,1),2*len(nCs)+len(nBs)-6)
            U[('o',1,s,0)]=net.unit(1,('o',s,0),[(IO(s),5)]+Eterms(NB[s],1),deg-7)
        U[('x',1,s,1-par(s))]=net.unit(1,('x',s,1-par(s)),[(IP(s),5)]+Eterms(NB[s],1),deg-5)
        U[('x',1,s,par(s))]=net.unit(1,('x',s,par(s)),[(IP(s),5)]+Eterms(NB[s],1),deg-6)
    # ---- L2..L4
    for l in (2,3,4):
        M=Ml[l]
        for s in range(81):
            U[('b',l,s)]=net.unit(l,('b',s),[(U[('b',l-1,s)],1.0)],0.0)
        for s in range(81):
            for k in (0,1,2):
                terms=[(U[('o',l-1,t,k)],1.0) for t in [s]+NB[s]]
                U[('o',l,s,k)]=net.unit(l,('o',s,k),terms+[(U[('b',l-1,s)],M)],-2*M)
            for k in (0,1):
                terms=[(U[('x',l-1,t,k)],1.0) for t in [s]+NB[s]]
                U[('x',l,s,k)]=net.unit(l,('x',s,k),terms+[(U[('b',l-1,s)],-M)],0.0)
    # ---- L5
    for s in range(81):
        U[('O',5,s)]=net.unit(5,('O',s),[(U[('b',4,s)],1.0)],-1.0)
    for s in BSUB:
        U[('b',5,s)]=net.unit(5,('b',s),[(U[('b',4,s)],1.0)],0.0)
    for q in range(81):
        for k in own_colors(q):
            terms=[(U[('o',4,t,k)],-2.0) for t in [q]+NB[q]]
            U[('m',5,q,k)]=net.unit(5,('m',q,k),terms+[(U[('b',4,q)],1.0)],-1.0)
        k=1-par(q)
        terms=[(U[('x',4,t,k)],1.0) for t in [q]+NB[q]]
        U[('v',5,q)]=net.unit(5,('v',q),terms+[(U[('b',4,q)],-3125.0)],0.0)
        k=par(q)
        terms=[(U[('x',4,t,k)],-2.0) for t in [q]+NB[q]]
        U[('n4',5,q)]=net.unit(5,('n4',q),terms+[(U[('b',4,q)],-5.0)],1.0)
    # ---- L6
    for q in range(81):
        terms=[(U[('n4',5,q)],2.0)]+[(U[('v',5,t)],-2.0) for t in NB[q]]
        U[('n5',6,q)]=net.unit(6,('n5',q),terms,-1.0)
        U[('at',6,q)]=net.unit(6,('at',q),[(U[('m',5,q,k)],1.0) for k in own_colors(q)],-1.0)
    for p in range(81):
        deg=len(NB[p])
        # n_e = O5'_p/eps - 1
        t1=[(U[('O',5,p)],-P['B1']/eps)]
        for q in NB[p]:
            t1.append((U[('O',5,q)],-P['BIG1']*2.0))
            for k in own_colors(q): t1.append((U[('m',5,q,k)],P['BIG1']))
            t1.append((U[('n4',5,q)],-P['BIGC']))
        U[('U1',6,p)]=net.unit(6,('U1',p),t1,P['B1']*3.0-0.2)
        t0=[(U[('O',5,p)],-P['B0']/eps)]
        for q in NB[p]:
            t0.append((U[('O',5,q)],-P['BIG0']))
            for k in own_colors(q): t0.append((U[('m',5,q,k)],P['BIG0']))
            t0.append((U[('n4',5,q)],-P['BIGC']))
        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*1.5-1.0)
        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))
        # positional aggregator
        r,c=divmod(p,9); tp=[]; bias=P['C']
        for t in range(81):
            dd=abs(t//9-r)+abs(t%9-c)
            if dd==0 or dd>3: continue
            wo=P['prox_own'].get(dd,0.0); wp=P['prox_opp'].get(dd,0.0)
            if t in BSUB:
                # wo*O + wp*P, P = 1 + O5' - b5
                tp+=[(U[('O',5,t)],wo+wp),(U[('b',5,t)],-wp)]; bias+=wp
            else:
                tp+=[(U[('O',5,t)],wo)]
        U[('pos',6,p)]=net.unit(6,('pos',p),tp,bias)
    # ---- output
    for p in range(81):
        terms=[(U[('n5',6,q)],P['CAP']) for q in NB[p]]
        terms+=[(U[('at',6,q)],P['ESC']) for q in NB[p]]
        terms+=[(U[('U1',6,p)],-P['SA1']),(U[('U0',6,p)],-P['SA0']),(U[('eye',6,p)],-P['EYE']),(U[('pos',6,p)],1.0)]
        net.outw(p,terms,-P['C']+P['LV'][line(p)])
    net.outw(81,[],P['PASS'])
    return net,U
