```python
from collections import deque
def seeds(b,col):
    s=set()
    for i in range(81):
        if b[i]!=col: continue
        e=sum(1 for j in go.NB[i] if b[j]==0)
        if e>=2: s.add(i)
        elif e==1 and any(b[j]==col and any(b[k]==0 for k in go.NB[j]) for j in go.NB[i]): s.add(i)
    return s
def chaindist(b,i,S):
    col=b[i]; dq=deque([(i,0)]); seen={i}
    while dq:
        x,dd=dq.popleft()
        if x in S: return dd
        for y in go.NB[x]:
            if b[y]==col and y not in seen: seen.add(y); dq.append((y,dd+1))
    return 99
cnt=Counter(); gs=Counter()
for p,(b,turn) in enumerate(boards):
    L=libs_map(b); h5=A[5][p]
    for col,base in ((turn,162),(3-turn,243)):
        S=seeds(b,col)
        for i in range(81):
            if b[i]==col and L[i]>=2 and h5[base+i]>0:
                cnt[min(chaindist(b,i,S),9)]+=1
                gs[min(len(go.group(b,i)[0]),30)//5]+=1
print(cnt); print(gs)
```