```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
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
st=collections.Counter()
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81 or k==0: continue
        prev=tob(B[gi,k-1]); p=m['move']
        r=G.play(prev,p,me)
        if r is None: continue
        nb,cap,_=r
        C5=prop(prev,3-me,seeds_par2(prev,3-me),5)
        judged=any(prev[q]==3-me and par(q) not in C5[q] for q in G.NB[p])
        st['moves']+=1
        if judged:
            st['judged_cap']+=1
            if not cap:
                st['FP_played']+=1
                g2,l2=G.group(nb,p)
                if len(l2)==1: st['FP_selfatari']+=1
print(st)
```