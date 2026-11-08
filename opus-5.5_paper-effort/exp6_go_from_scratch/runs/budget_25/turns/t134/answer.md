```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_23/games.json')); B=np.load('/work/pass_23/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
par=lambda s:(s//9+s%9)%2
def p3(s):
    r,c=divmod(s,9)
    if (r+c)%2==0: return 0
    return 1 if r%2==0 else 2
def seeds(board,side):
    S={}
    for s in range(81):
        if board[s]!=side: continue
        libs=[n for n in G.NB[s] if board[n]==0]; m=len(libs); st=set()
        if par(s)==1:
            if m>=1: st.add(0)
            if m>=2: st.add(1)
            if m>=3: st.add(2)
        else:
            nB=sum(1 for n in libs if p3(n)==1); nC=m-nB
            if 2*nB+nC>=2: st.add(1)
            if 2*nC+nB>=2: st.add(2)
            if m>=3: st.add(0)
        S[s]=st
    return S
def prop(board,side,S,R):
    cur=S
    for _ in range(R):
        new={}
        for s in cur:
            st=set(cur[s])
            for n in G.NB[s]:
                if board[n]==side: st|=cur[n]
            new[s]=st
        cur=new
    return cur
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
for gi in (44,64,57):
    g=d[gi]; n=len(g['moves']); last=B[gi,n-1]; me=1 if g['net_color']=='black' else 2
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    ply=match+1; prev=tob(B[gi,ply-1]); mv=g['moves'][ply]['move']
    Co=prop(prev,me,seeds(prev,me),4)
    own=[q for q in G.NB[mv] if prev[q]==me]
    print(gi,divmod(mv,9),'own nbrs',[(divmod(q,9),len(G.group(prev,q)[1]),sorted(Co[q])) for q in own],'ne',sum(1 for x in G.NB[mv] if prev[x]==0), 'top5',g['moves'][ply]['top5'][:3])
    print(showme(prev,me))
```