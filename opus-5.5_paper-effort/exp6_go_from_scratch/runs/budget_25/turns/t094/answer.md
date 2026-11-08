```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
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
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter(); prs=collections.defaultdict(list)
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
            ne=sum(1 for n in G.NB[p] if prev[n]==0); own=[n for n in G.NB[p] if prev[n]==me]
            if ne!=1 or len(own)!=1: continue
            Co=prop(prev,me,seeds(prev,me),4)
            q=own[0]; gg,ll=G.group(prev,q)
            e=[n for n in G.NB[p] if prev[n]==0][0]
            key=('truelibs',len(ll),'colors',len(Co[q]),'otherlib==e' if e in ll else 'other')
            cat[key]+=1; prs[key].append(dict(m['top5']).get(p,0))
for k,v in cat.most_common(10): print(k,v,'meanprob',round(np.mean(prs[k]),2))
```