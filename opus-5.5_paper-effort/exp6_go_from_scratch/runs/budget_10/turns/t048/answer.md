```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
B=np.concatenate([np.load(f'/work/pass_{k}/boards.npy') for k in (2,4,5)]); 
V=np.concatenate([np.load(f'/work/pass_{k}/valid.npy') for k in (2,4,5)])
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
res=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        opp=3-col
        flags={}
        for c in (col,opp):
            e={i:E(b,i) for i in range(81) if b[i]==c}
            sig={i:sum(e[j] for j in go.NB[i] if b[j]==c) for i in e}
            Fs=set(i for i in e if e[i]>=2 or (e[i]==1 and any(b[j]==c and e[j]>=1 for j in go.NB[i])))
            F3s=set(i for i in e if e[i]+sig[i]>=3)
            Fn=set(i for i in e if 2.5*e[i]+sig[i]-2.5>0)
            F3n=set(i for i in e if 2*e[i]+sig[i]-4.5>0)
            flags[c]={'cur':(Fs,F3s),'new':(Fn,F3n)}
        for p in range(81):
            if b[p]!=0: continue
            ep=E(b,p)
            if ep>1: continue
            if not any(b[j]==col for j in go.NB[p]) and ep==0: continue
            # true result
            gg=go.Game(); gg.b=b.copy(); gg.turn=col
            if not gg.legal(p): continue
            gg.play(p); Gp,Lp=go.group(gg.b,p)
            trueSA=len(Lp)<=1
            for name in ('cur','new'):
                F,F3=flags[col][name]; Fo,F3o=flags[opp][name]
                n3=any(b[j]==col and within(b,j,F3) for j in go.NB[p])
                capt=any(b[j]==opp and not within(b,j,Fo|F3o) for j in go.NB[p])
                pred=(not n3) and (not capt)
                res[(name,trueSA,pred)]+=1
for kk in sorted(res,key=str): print(kk,res[kk])
```