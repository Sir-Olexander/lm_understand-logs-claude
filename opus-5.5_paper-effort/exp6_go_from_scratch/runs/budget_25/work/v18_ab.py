
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
SKIPP=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44, 62,10,16,64])
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
    U[('gS',1)]=net.unit(1,'gS',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0'])
    U[('gD',1)]=net.unit(1,'gD',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0'])
    U[('gS2',1)]=net.unit(1,'gS2',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0']-P['Scap'])
    U[('gD2',1)]=net.unit(1,'gD2',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0']-P['Dcap'])
    for l in (2,3,4,5,6):
        for g in ('gS2','gD2'):
            U[(g,l)]=net.unit(l,g,[(U[(g,l-1)],1.0)],0.0)
    for l in (2,3,4,5):
        for g in ('gS','gD'):
            U[(g,l)]=net.unit(l,g,[(U[(g,l-1)],1.0)],0.0)
    # edge certification units (opp): three opp stones in a row on 2nd line, end stones' 1st-line points empty
    CERT=[]
    for c in P['cert_cs']:
        for (a,b2,l1,l2) in [((1,c),(1,c+1),(0,c),(0,c+2)),((7,c),(7,c+1),(8,c),(8,c+2)),((c,1),(c+1,1),(c,0),(c+2,0)),((c,7),(c+1,7),(c,8),(c+2,8))]:
            pass
    for c in P['cert_cs']:
        trip=[[(1,c),(1,c+1),(1,c+2),(0,c),(0,c+2)],[(7,c),(7,c+1),(7,c+2),(8,c),(8,c+2)],[(c,1),(c+1,1),(c+2,1),(c,0),(c+2,0)],[(c,7),(c+1,7),(c+2,7),(c,8),(c+2,8)]]
        for t in trip:
            st=[r*9+cc for r,cc in t[:3]]; lb=[r*9+cc for r,cc in t[3:]]
            terms=[(IP(s),3.0) for s in st]+[(IO(x),-1.0) for x in lb]+[(IP(x),-1.0) for x in lb]
            j=net.unit(1,('cert',st[1]),terms,2.0-10.0)
            CERT.append((st[1],par(st[0]),j))
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
                if l==2: terms+=[(j,1.0) for (sm,kk,j) in CERT if sm==s and kk==k]
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
        # combined self-atari unit Uc = relu(a - b*n_e - c2*sum W - c1*k - BIGC*sum n4)
        a,b_,c2,c1=P['Ua'],P['Ub'],P['Uw'],P['Uk']
        tu=[(U[('O',5,p)],-b_/eps)]; bias=a+b_
        for q in NB[p]:
            # W_q = O_q - sum m ; k = sum O_q
            tu.append((U[('O',5,q)],-c2-c1))
            for k in own_colors(q): tu.append((U[('m',5,q,k)],c2))
            tu.append((U[('n4',5,q)],-P['BIGC']))
        U[('Uc',6,p)]=net.unit(6,('Uc',p),tu,bias-0.3)
        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))
        if p in P['UXPTS']:
            tx=[(U[('O',5,p)],-1.0/eps)]
            for q in NB[p]:
                for k in own_colors(q): tx.append((U[('m',5,q,k)],1.0))
            U[('Ux',6,p)]=net.unit(6,('Ux',p),tx,1.0-0.5-0.1)
        # positional: A_o, A_x
        r,c=divmod(p,9); Ao=[]; Ax=[]; axb=0.0
        for t in range(81):
            dd=abs(t//9-r)+abs(t%9-c)
            if dd==0 or dd>3 or t not in BSUB and False: pass
            if dd==0 or dd>3: continue
            w=P['kern'][dd]
            Ao.append((U[('O',5,t)],w))
            if t in BSUB:
                Ax+=[(U[('O',5,t)],w),(U[('b',5,t)],-w)]; axb+=w
        lam,mu=P['lam'],P['mu']
        t1=[(i,(lam-mu)*w) for i,w in Ax]+[(i,-mu*w) for i,w in Ao]
        t1.append((U[('O',5,p)],P['eta']/eps))
        U[('P1',6,p)]=net.unit(6,('P1',p),t1,P['C']+(lam-mu)*axb-P['eta'])
        t2=[(i,w) for i,w in Ax]+[(i,-w) for i,w in Ao]
        U[('P2',6,p)]=net.unit(6,('P2',p),t2,axb)
    U[('gS',6)]=net.unit(6,'gS',[(U[('gS',5)],1.0)],0.0)
    U[('gD',6)]=net.unit(6,'gD',[(U[('gD',5)],1.0)],0.0)
    # ---- output
    for p in range(81):
        terms=[(U[('n5',6,q)],P['CAP']) for q in NB[p]]
        terms+=[(U[('at',6,q)],P['ESC']) for q in NB[p]]
        terms+=[(U[('Uc',6,p)],-P['SAc']),(U[('eye',6,p)],-P['EYE']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]
        if ('Ux',6,p) in U: terms.append((U[('Ux',6,p)],-P['WX']))
        terms.append((U[('gS',6)],P['kL'].get(line(p),0.0)))
        net.outw(p,terms,-P['C']+P['LV'][line(p)])
    net.outw(81,[(U[('gS',6)],P['kS']),(U[('gD',6)],P['kD']),(U[('gS2',6)],-P['kS']),(U[('gD2',6)],-P['kD'])],P['PASS'])
    return net,U
