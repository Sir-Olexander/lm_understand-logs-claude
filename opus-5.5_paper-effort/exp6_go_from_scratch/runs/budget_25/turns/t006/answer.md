```python
import sys; sys.path.insert(0,'/work')
import importlib, netlib_ab as NL; importlib.reload(NL)
import numpy as np
NB,EDGES,IO,IP=NL.NB,NL.EDGES,NL.IO,NL.IP
net=NL.Net()
L=1
U={}
# L1 local features
for s in range(81):
    deg=len(NB[s])
    # empty
    U[('E',s)]=net.unit(1,('E',s),[(IO(s),-1),(IP(s),-1)],1.0)
    # own eye-like: all nbrs own, s empty
    U[('eye',s)]=net.unit(1,('eye',s),[(IO(n),1) for n in NB[s]]+[(IO(s),-2),(IP(s),-2)],-(deg-1))
    # opp single stone in atari
    U[('xat1',s)]=net.unit(1,('xat1',s),[(IP(s),2)]+[(IO(n),1) for n in NB[s]]+[(IP(n),-2) for n in NB[s]],-deg)
    U[('oat1',s)]=net.unit(1,('oat1',s),[(IO(s),2)]+[(IP(n),1) for n in NB[s]]+[(IO(n),-2) for n in NB[s]],-deg)
    # opp single stone with <=2 libs (value 2 if atari, 1 if 2 libs)
    U[('xle2',s)]=net.unit(1,('xle2',s),[(IP(s),2)]+[(IO(n),1) for n in NB[s]]+[(IP(n),-2) for n in NB[s]],-deg+1)
    # single-stone self-atari for own move at s: no own nbr, exactly 1 empty nbr  (value 2 if 0 empty)
    U[('sa1',s)]=net.unit(1,('sa1',s),[(IP(n),1) for n in NB[s]]+[(IO(n),-1) for n in NB[s]]+[(IO(s),-3),(IP(s),-3)],2-deg)
for (s,t) in EDGES:
    Nst=[x for x in set(NB[s])|set(NB[t]) if x not in (s,t)]
    m=len(Nst)
    U[('xdom',s,t)]=net.unit(1,('xdom',s,t),[(IP(s),2),(IP(t),2)]+[(IO(x),1) for x in Nst]+[(IP(x),-2) for x in Nst],-(m+2))
    U[('odom',s,t)]=net.unit(1,('odom',s,t),[(IO(s),2),(IO(t),2)]+[(IP(x),1) for x in Nst]+[(IO(x),-2) for x in Nst],-(m+2))
print('L1 used',net.n[0])
C=200.0
EYE=40; CAP1=12; CAP2=16; ESC1=8; ESC2=10; ATK1=2.0; LIB=0.5; SA1=6
# L2 accumulator
for p in range(81):
    terms=[(U[('eye',p)],-EYE),(U[('sa1',p)],-SA1)]
    for q in NB[p]:
        terms+= [(U[('xat1',q)],CAP1),(U[('oat1',q)],ESC1),(U[('xle2',q)],ATK1),(U[('xat1',q)],-2*ATK1),(U[('E',q)],LIB)]
    for (s,t) in EDGES:
        Nst=set(NB[s])|set(NB[t])
        if p in Nst and p not in (s,t):
            terms+=[(U[('xdom',s,t)],CAP2),(U[('odom',s,t)],ESC2)]
    U[('acc',2,p)]=net.unit(2,('acc',p),terms,C)
for l in range(3,7):
    for p in range(81):
        U[('acc',l,p)]=net.unit(l,('acc',p),[(U[('acc',l-1,p)],1.0)],0.0)
# output
def line(s):
    r,c=divmod(s,9); return min(r,c,8-r,8-c)+1
LV={1:-2.0,2:0.0,3:1.5,4:1.5,5:1.0}
for p in range(81):
    net.outw(p,[(U[('acc',6,p)],1.0)],-C+LV[line(p)])
net.outw(81,[],-15.0)
net.save_all(save)
print(saved())
import pickle; pickle.dump({k:v for k,v in U.items()},open('/work/v1_units.pkl','wb'))
```