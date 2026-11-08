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
res={1:collections.Counter(),-1:collections.Counter()}
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2; st=res[g['result']]
    st['games']+=1
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or k==0: continue
        prev=tob(B[gi,k-1]); st['moves']+=1
        if m['move']==81: st['pass']+=1; continue
        r=G.play(prev,m['move'],me)
        if r is None: continue
        nb,cap,_=r; g2,l2=G.group(nb,m['move'])
        C5=prop(prev,3-me,seeds2(prev,3-me),5)
        judged=any(prev[q]==3-me and par(q) not in C5[q] for q in G.NB[m['move']])
        if judged and not cap: st['FPcap']+=1
        if len(l2)==1 and not cap: st['selfatari']+=1; st['sa_stones']+=len(g2)
        if cap: st['captures']+=1; st['cap_stones']+=len(cap)
        ranks=[a for a,p in m['top5']]
        if m['move'] not in ranks[:1]: st['not_top1']+=1
for r,st in res.items():
    gn=st['games']; print(r,{k:round(v/gn,2) for k,v in st.items()})
```