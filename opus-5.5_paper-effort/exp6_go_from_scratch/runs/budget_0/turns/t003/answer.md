```python
import numpy as np, json, pickle

PTS = [(r,c) for r in range(9) for c in range(9)]
def nb(q):
    r,c=q
    return [(r+dr,c+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)) if 0<=r+dr<9 and 0<=c+dc<9]
def diag(q):
    r,c=q
    return [(r+dr,c+dc) for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)) if 0<=r+dr<9 and 0<=c+dc<9]
def jump(q):
    r,c=q
    return [(r+dr,c+dc) for dr,dc in ((-2,0),(2,0),(0,-2),(0,2)) if 0<=r+dr<9 and 0<=c+dc<9]
def d12(q):
    return nb(q)+diag(q)+jump(q)
def deg(q): return len(nb(q))
def line(q):
    r,c=q; return min(r,c,8-r,8-c)

class Layer:
    def __init__(self, name):
        self.name=name; self.units=[]; self.idx={}
    def add(self, key, bias, w):
        assert key not in self.idx, key
        self.idx[key]=len(self.units)
        self.units.append((key,float(bias),dict(w)))
def W(d,k,v):
    d[k]=d.get(k,0.0)+v

layers=[]
# ---------- L1 (inputs: raw x) ----------
L1=Layer('L1')
for q in PTS:
    L1.add(('own',q),0,{('xo',q):1})
    L1.add(('opp',q),0,{('xp',q):1})
for col,xc in (('own','xo'),('opp','xp')):
    for q in PTS:
        w={}; W(w,(xc,q),1)
        for n in nb(q): W(w,('xo',n),1); W(w,('xp',n),1)
        L1.add(('Z'+col,q),-deg(q),w)          # relu(c_q - E_q) = [c_q and E_q==0]
    for q in PTS:
        w={}; W(w,(xc,q),5)
        for n in nb(q): W(w,('xo',n),-1); W(w,('xp',n),-1)
        L1.add(('E'+col,q),deg(q)-5,w)         # E_q if c_q else 0
layers.append(L1)

# ---------- L2 ----------
L2=Layer('L2')
for q in PTS:
    L2.add(('own',q),0,{('own',q):1}); L2.add(('opp',q),0,{('opp',q):1})
for col,oth in (('own','opp'),('opp','own')):
    for q in PTS:   # U-seed: c_q & E>=1 & (E>=2 or same-colour nbr with lib)
        w={}; W(w,(col,q),9); W(w,('Z'+col,q),-9)
        for n in nb(q): W(w,(oth,n),-1); W(w,('Z'+col,n),-1)
        L2.add(('S'+col,q),deg(q)-1-9,w)
    for q in PTS:   # V-seed: c_q & E>=1 & 8E+5s>=18  (=> group has >=3 libs)
        w={}; W(w,(col,q),120); W(w,('Z'+col,q),-120)
        for n in nb(q):
            W(w,('own',n),-8); W(w,('opp',n),-8); W(w,('E'+col,n),5)
        L2.add(('VS'+col,q),8*deg(q)-17-120,w)
layers.append(L2)

# ---------- L3 ----------
L3=Layer('L3')
for q in PTS:
    L3.add(('own',q),0,{('own',q):1}); L3.add(('opp',q),0,{('opp',q):1})
for col in ('own','opp'):
    for q in PTS:
        w={}; W(w,(col,q),1); W(w,('S'+col,q),-1)
        for n in nb(q): W(w,('S'+col,n),-1)
        L3.add(('U1'+col,q),0,w)
    for q in PTS:
        w={}; W(w,(col,q),1); W(w,('VS'+col,q),-1)
        for n in nb(q): W(w,('VS'+col,n),-1)
        L3.add(('NV1'+col,q),0,w)
layers.append(L3)

# ---------- L4 ----------
L4=Layer('L4')
for q in PTS:
    L4.add(('own',q),0,{('own',q):1}); L4.add(('opp',q),0,{('opp',q):1})
for col in ('own','opp'):
    for pre,new in (('U1','U2'),('NV1','NV2')):
        for q in PTS:
            w={}; W(w,(pre+col,q),1)
            for n in nb(q): W(w,(col,n),-1); W(w,(pre+col,n),1)
            L4.add((new+col,q),0,w)
layers.append(L4)

# ---------- L5 ----------
L5=Layer('L5')
for q in PTS:
    L5.add(('own',q),0,{('own',q):1}); L5.add(('opp',q),0,{('opp',q):1})
for col,pre,new in (('own','U2','U3'),('opp','U2','U3'),('opp','NV2','NV3')):
    for q in PTS:
        w={}; W(w,(pre+col,q),1)
        for n in nb(q): W(w,(col,n),-1); W(w,(pre+col,n),1)
        L5.add((new+col,q),0,w)
for p in PTS:   # SOFT = relu(2 - E_p - 2occ - 2*#adj unsafe opp - 2*#adj very-safe own)
    w={}; W(w,('own',p),-2); W(w,('opp',p),-2)
    for n in nb(p):
        W(w,('own',n),-1); W(w,('NV2own',n),2); W(w,('opp',n),1); W(w,('U2opp',n),-2)
    L5.add(('SOFT',p),2-deg(p),w)
layers.append(L5)

# ---------- L6 ----------
C=300.0
PRIOR={0:-8.0,1:0.0,2:6.0,3:8.0,4:14.0}
L6=Layer('L6')
for p in PTS:
    w={}
    for n in nb(p):
        W(w,('own',n),-3+1); W(w,('opp',n),-3+5)
        W(w,('U3opp',n),80); W(w,('NV3opp',n),12)
    for d in diag(p): W(w,('own',d),2); W(w,('opp',d),1)
    for j in jump(p): W(w,('own',j),1); W(w,('opp',j),1)
    W(w,('SOFT',p),-50)
    L6.add(('S',p),C+PRIOR[line(p)]+3*deg(p),w)
for p in PTS:
    w={}; W(w,('SOFT',p),-4)
    for n in nb(p): W(w,('U3own',n),1)
    L6.add(('ESCX',p),0,w)
for p in PTS:
    w={}; W(w,('own',p),-2); W(w,('opp',p),-2)
    for n in nb(p):
        W(w,('own',n),-1); W(w,('opp',n),1); W(w,('U3opp',n),-2); W(w,('U3own',n),2)
    L6.add(('HARD',p),2-deg(p),w)
for p in PTS:
    w={}; W(w,('own',p),-4); W(w,('opp',p),-4)
    for n in nb(p): W(w,('own',n),1)
    L6.add(('EYE',p),1-deg(p),w)
for p in PTS:
    w={}; W(w,('own',p),-4); W(w,('opp',p),-4)
    for n in nb(p): W(w,('own',n),1); W(w,('U3own',n),-4)
    for q in d12(p): W(w,('opp',q),-4)
    L6.add(('IN',p),0,w)
for p in PTS:
    w={}; W(w,('own',p),-4); W(w,('opp',p),-4)
    for n in nb(p): W(w,('opp',n),1)
    L6.add(('ALLOPP',p),1-deg(p),w)
layers.append(L6)

OUTW={'S':1.0,'ESCX':100.0,'HARD':-40.0,'EYE':-300.0,'IN':-150.0,'ALLOPP':250.0}
PASS_LOGIT=-20.0

for L in layers: print(L.name, len(L.units))
pickle.dump({'layers':[(L.name,L.units) for L in layers],'C':C,'OUTW':OUTW,'PASS':PASS_LOGIT},open('/work/spec.pkl','wb'))
```