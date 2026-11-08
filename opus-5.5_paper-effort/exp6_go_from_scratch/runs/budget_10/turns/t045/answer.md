```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
B=np.concatenate([np.load(f'/work/pass_{k}/boards.npy') for k in (2,4,5)]); 
V=np.concatenate([np.load(f'/work/pass_{k}/valid.npy') for k in (2,4,5)])
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
res=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        e={i:E(b,i) for i in range(81) if b[i]==col}
        Fs=set(i for i in e if e[i]>=2 or (e[i]==1 and any(b[j]==col and e[j]>=1 for j in go.NB[i])))
        F3s=set(i for i in e if e[i]+sum(e[j] for j in go.NB[i] if b[j]==col)>=3)
        # edge-based
        Es=set(i for i in e if any(b[j]==col and e[i]+e[j]>=2 for j in go.NB[i]))
        E2=set(i for i in e if e[i]>=2)
        F3n=set(i for i in e if sum(max(e[i]+e[j]-1,0) for j in go.NB[i] if b[j]==col)+1.2*e[i]-3.4>0)
        for q in e:
            if e[q]==0: continue
            G_,L=go.group(b,q); nl=min(len(L),3)
            cur= dist(b,q,Fs|F3s)<=3
            new= dist(b,q,Es)<=4 or dist(b,q,E2)<=3 or dist(b,q,F3n)<=3
            curT= dist(b,q,F3s)<=3; newT=dist(b,q,F3n)<=3
            res[('A',nl,'cur',cur)]+=1; res[('A',nl,'new',new)]+=1
            res[('T',nl,'cur',curT)]+=1; res[('T',nl,'new',newT)]+=1
for kk in sorted(res,key=str): print(kk,res[kk])
```