
import numpy as np, pickle, sys
sys.path.insert(0,'/work')
import go
from collections import Counter, deque
def load_pass(k):
    d=f'/work/pass_{k}/'
    out={'X':np.load(d+'inputs.npy'),'logits':np.load(d+'logits.npy'),'probs':np.load(d+'probs.npy'),'legal':np.load(d+'legal.npy')}
    import os
    for l in range(1,7):
        f=d+f'act_L{l}.npy'
        if os.path.exists(f): out[l]=np.load(f)
    return out
def libs_map(b):
    L=np.zeros(81,int)
    for i in range(81):
        if b[i]: L[i]=len(go.group(b,i)[1])
    return L
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
def rc(i): return divmod(i,9)
