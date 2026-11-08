```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
# sample positions
rng=np.random.default_rng(0)
idx=[(g,k) for g in range(B.shape[0]) for k in range(B.shape[1]) if V[g,k] and k%3==0]
print(len(idx))
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
hashes={'r2':lambda p:(p//9)%2,'c2':lambda p:(p%9)%2,'rc2':lambda p:((p//9)+(p%9))%2,
        'r2c2xor':lambda p:((p//9)//2+(p%9)//2)%2, 'mod5':lambda p:int(((p//9)+2*(p%9))%5<2)}
res=Counter()
for (g,k) in idx:
    b=conv(B[g,k])
    for col in (1,2):
        e={i:E(b,i) for i in range(81) if b[i]==col}
        Fs=set(i for i in e if e[i]>=2 or (e[i]==1 and any(b[j]==col and e[j]>=1 for j in go.NB[i])))
        F3s=set(i for i in e if e[i]+sum(e[j] for j in go.NB[i] if b[j]==col)>=3)
        Qs={h:(set(),set()) for h in hashes}
        for i in e:
            for h,f in hashes.items():
                for j in go.NB[i]:
                    if b[j]==0: Qs[h][f(j)].add(i)
                if i in Fs or i in F3s: Qs[h][0].add(i); Qs[h][1].add(i)
        done=set()
        for i in e:
            if i in done: continue
            G,L=go.group(b,i); done|=G
            nl=min(len(L),3)
            # relevant stones: those adjacent to a liberty
            for q in G:
                if e[q]==0: continue
                aA=not within(b,q,Fs|F3s); tA=not within(b,q,F3s)
                res[('A',nl,aA)]+=1; res[('T',nl,tA)]+=1
                for h in hashes:
                    aB=not (within(b,q,Qs[h][0]) and within(b,q,Qs[h][1]))
                    res[(h,nl,aB)]+=1
for key in ['A','T']+list(hashes):
    print(key,[ (nl, res[(key,nl,True)], res[(key,nl,False)]) for nl in (1,2,3)])
```