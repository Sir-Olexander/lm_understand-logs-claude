```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_20/games.json')); B=np.load('/work/pass_20/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
par=lambda s:(s//9+s%9)%2
def seeds2(board,side):
    S={}
    for s in range(81):
        if board[s]!=side: continue
        m=sum(1 for n in G.NB[s] if board[n]==0); st=set()
        if m>=1: st.add(1-par(s))
        if m>=2: st.add(par(s))
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
for gi in (0,9,10,89,125,126):
    g=d[gi]; n=len(g['moves']); last=B[gi,n-1]; me=1 if g['net_color']=='black' else 2
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    ply=match+1; prev=tob(B[gi,ply-1]); mv=g['moves'][ply]['move']
    C4=prop(prev,3-me,seeds2(prev,3-me),4); C5=prop(prev,3-me,seeds2(prev,3-me),5)
    for q in G.NB[mv]:
        if prev[q]==3-me:
            gg,ll=G.group(prev,q)
            print(gi,divmod(mv,9),'opp',divmod(q,9),'size',len(gg),'libs',sorted(divmod(x,9) for x in ll),'R4',par(q) not in C4[q],'R5',par(q) not in C5[q])
```