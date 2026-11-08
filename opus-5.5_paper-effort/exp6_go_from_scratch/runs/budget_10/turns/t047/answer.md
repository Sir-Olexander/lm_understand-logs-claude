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
res=Counter(); bad=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        e={i:E(b,i) for i in range(81) if b[i]==col}
        sig={i:sum(e[j] for j in go.NB[i] if b[j]==col) for i in e}
        Fs=set(i for i in e if e[i]>=2 or (e[i]==1 and any(b[j]==col and e[j]>=1 for j in go.NB[i])))
        F3s=set(i for i in e if e[i]+sig[i]>=3)
        Fn=set(i for i in e if 2.5*e[i]+sig[i]-2.5>0)
        F3n=set(i for i in e if 2*e[i]+sig[i]-4.5>0)
        # exactness check of new seeds
        for i in Fn:
            if len(go.group(b,i)[1])<2: bad['F']+=1
        for i in F3n:
            if len(go.group(b,i)[1])<3: bad['F3']+=1
        for q in e:
            if e[q]==0: continue
            G_,L=go.group(b,q); nl=min(len(L),3)
            for name,FF,FF3 in (('cur',Fs,F3s),('new',Fn,F3n)):
                res[('A',nl,name,dist(b,q,FF|FF3)<=3)]+=1
                res[('T',nl,name,dist(b,q,FF3)<=3)]+=1
print(bad)
for kk in sorted(res,key=str): print(kk,res[kk])
```