```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
B=np.concatenate([np.load(f'/work/pass_{k}/boards.npy') for k in (8,9)]); 
V=np.concatenate([np.load(f'/work/pass_{k}/valid.npy') for k in (8,9)])
idx=[(g,k) for g in range(B.shape[0]) for k in range(B.shape[1]) if V[g,k] and k%3==0]
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
E=lambda b,i: sum(1 for j in go.NB[i] if b[j]==0)
def within(b,i,S,R=3):
    col=b[i]; dq=deque([(i,0)]); seen={i}
    while dq:
        x,dd=dq.popleft()
        if x in S: return True
        if dd==R: continue
        for y in go.NB[x]:
            if b[y]==col and y not in seen: seen.add(y); dq.append((y,dd+1))
    return False
R2={i:[j for j in range(81) if 0<abs(i//9-j//9)+abs(i%9-j%9)<=2] for i in range(81)}
rules={'exact':lambda e,s:2*e+s-4.5>0,'old3':lambda e,s:e+s>=3,'mix':lambda e,s:1.5*e+s>=4}
lams=[0,0.15,0.25]
res=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        opp=3-col
        e={i:E(b,i) for i in range(81) if b[i]==col}
        sig={i:sum(e[j] for j in go.NB[i] if b[j]==col) for i in e}
        eo={i:E(b,i) for i in range(81) if b[i]==opp}
        sigo={i:sum(eo[j] for j in go.NB[i] if b[j]==opp) for i in eo}
        Fo=set(i for i in eo if 2.5*eo[i]+sigo[i]-2.5>0); F3o=set(i for i in eo if 2*eo[i]+sigo[i]-4.5>0)
        F3={r:set(i for i in e if f(e[i],sig[i])) for r,f in rules.items()}
        for p in range(81):
            if b[p]!=0: continue
            ep=E(b,p)
            if ep>1: continue
            if not any(b[j]==col for j in go.NB[p]): continue
            gg=go.Game(); gg.b=b.copy(); gg.turn=col
            if not gg.legal(p): continue
            gg.play(p); Gp,Lp=go.group(gg.b,p); trueSA=len(Lp)<=1
            nA=sum(1 for j in go.NB[p] if b[j]==opp and not within(b,j,Fo|F3o))
            dens=sum(1 for j in R2[p] if b[j]==col)
            for r in rules:
                n3=sum(1 for j in go.NB[p] if b[j]==col and within(b,j,F3[r]))
                for lam in lams:
                    v=max(0,2-ep-2*n3-2*nA-lam*dens)
                    res[(r,lam,trueSA)]+=v
                    res[(r,lam,trueSA,'n')]+=1
for r in rules:
    for lam in lams:
        print(r,lam,'SA mass on trueSA',round(res[(r,lam,True)]/res[(r,lam,True,'n')],3),' on non-SA',round(res[(r,lam,False)]/res[(r,lam,False,'n')],3))
```