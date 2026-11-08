```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
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
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter(); szs=collections.Counter()
for ps in (13,14):
    d=json.load(open('/work/pass_%d/games.json'%ps)); B=np.load('/work/pass_%d/boards.npy'%ps)
    for gi,g in enumerate(d):
        if g['opponent']!='strong': continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            prev=tob(B[gi,k-1]); p=m['move']
            r=G.play(prev,p,me)
            if r is None: continue
            g2,l2=G.group(r[0],p)
            if len(l2)!=1 or r[1]: continue
            C4=prop(prev,3-me,seeds2(prev,3-me),4); C5=prop(prev,3-me,seeds2(prev,3-me),5)
            opp=[q for q in G.NB[p] if prev[q]==3-me]
            f4=any(par(q) not in C4[q] for q in opp); f5=any(par(q) not in C5[q] for q in opp)
            # collision vs radius: R=20
            C20=prop(prev,3-me,seeds2(prev,3-me),20); f20=any(par(q) not in C20[q] for q in opp)
            ne=sum(1 for n in G.NB[p] if prev[n]==0); kown=sum(1 for n in G.NB[p] if prev[n]==me)
            key=('R4fp' if f4 else 'noR4', 'R5fp' if f5 else '-', 'collision' if f20 else ('radius' if f4 else '-'), 'k%d'%kown, 'ne%d'%ne)
            cat[key]+=1; szs[key]+=len(g2)
for k,v in cat.most_common(15): print(k,v,'stones',szs[k])
```