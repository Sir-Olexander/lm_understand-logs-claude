```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
par=lambda s:(s//9+s%9)%2
def seeds_par2(board,side):
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
for gi,k,p in [(0,60,80),(3,76,8),(7,40,26),(22,58,2)]:
    g=d[gi]; me=1 if g['net_color']=='black' else 2
    prev=tob(B[gi,k-1])
    C5=prop(prev,3-me,seeds_par2(prev,3-me),5)
    C4=prop(prev,3-me,seeds_par2(prev,3-me),4)
    for q in G.NB[p]:
        if prev[q]==3-me:
            gg,ll=G.group(prev,q)
            print(gi,k,divmod(p,9),'q',divmod(q,9),'par(q)',par(q),'C4',C4[q],'C5',C5[q],'libs',sorted(divmod(x,9) for x in ll)[:8], 'lib parities',sorted(set(par(x) for x in ll)))
```