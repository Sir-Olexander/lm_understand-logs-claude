```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
B=np.concatenate([np.load('/work/pass_2/boards.npy'),np.load('/work/pass_4/boards.npy')]); 
V=np.concatenate([np.load('/work/pass_2/valid.npy'),np.load('/work/pass_4/valid.npy')])
idx=[(g,k) for g in range(B.shape[0]) for k in range(B.shape[1]) if V[g,k] and k%2==0]
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
E=lambda b,i: sum(1 for j in go.NB[i] if b[j]==0)
def dist(b,i,S):
    col=b[i]; dq=deque([(i,0)]); seen={i}
    while dq:
        x,dd=dq.popleft()
        if x in S: return dd
        for y in go.NB[x]:
            if b[y]==col and y not in seen: seen.add(y); dq.append((y,dd+1))
    return 99
R2={i:[j for j in range(81) if 0<abs(i//9-j//9)+abs(i%9-j%9)<=2] for i in range(81)}
res=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        e={i:E(b,i) for i in range(81) if b[i]==col}
        Fs=set(i for i in e if e[i]>=2 or (e[i]==1 and any(b[j]==col and e[j]>=1 for j in go.NB[i])))
        F3s=set(i for i in e if e[i]+sum(e[j] for j in go.NB[i] if b[j]==col)>=3)
        S=Fs|F3s
        for q in e:
            if e[q]==0: continue
            if dist(b,q,S)<=3: continue
            G_,L=go.group(b,q)
            dens=sum(1 for j in R2[q] if b[j]==col)
            res[(len(L)==1, min(dens,10))]+=1
for t in (True,False):
    print('true' if t else 'false',[res[(t,d)] for d in range(11)])
```