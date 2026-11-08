```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_10/games.json'))
B=np.load('/work/pass_10/boards.npy')
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
for rad in (1,2):
    T=collections.Counter(); F=collections.Counter(); TB=collections.Counter()
    for gi,g in enumerate(d):
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or k==0: continue
            prev=tob(B[gi,k-1]); opp=3-me
            C4=prop(prev,opp,seeds2(prev,opp),4)
            gid,groups=G.all_groups(prev)
            for q in range(81):
                if prev[q]!=opp or par(q) in C4[q]: continue
                if not any(prev[n]==0 for n in G.NB[q]): continue
                gg,ll,c=groups[gid[q]]
                r,c_=divmod(q,9)
                I=sum((prev[t]==me)-(prev[t]==opp) for t in range(81) if abs(t//9-r)+abs(t%9-c_)<=rad)
                if len(ll)==1:
                    T[I]+=1
                    if len(gg)>=5: TB[I]+=1
                else: F[I]+=1
    print('rad',rad); print(' true',sorted(T.items())); print(' trueBig',sorted(TB.items())); print(' FP',sorted(F.items()))
```