```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def tob(b): return [1 if x==1 else (2 if x==-1 else 0) for x in b]
par=lambda s:(s//9+s%9)%2
def p3(s):
    r,c=divmod(s,9)
    if (r+c)%2==0: return 0
    return 1 if r%2==0 else 2
def mod3(s): return (s//9+s%9)%3
def seeds(board,side,scheme):
    S={}
    for s in range(81):
        if board[s]!=side: continue
        libs=[n for n in G.NB[s] if board[n]==0]; m=len(libs)
        st=set()
        if scheme=='par2':
            if m>=1: st.add(1-par(s))
            if m>=2: st.add(par(s))
        elif scheme=='p3':
            if par(s)==1:  # odd stone: libs even -> color 0
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
positions=[]
for gi,g in enumerate(d):
    for k in range(1,len(g['moves'])):
        positions.append((tob(B[gi,k-1]), 1 if k%2==0 else 2))
res=collections.Counter()
for board,me in positions[::3]:
    opp=3-me
    gid,groups=G.all_groups(board)
    tl={s:len(groups[gid[s]][1]) for s in range(81) if board[s]}
    Co={R:prop(board,me,seeds(board,me,'p3'),R) for R in (3,4)}
    Cx={R:prop(board,opp,seeds(board,opp,'par2'),R) for R in (3,4,5)}
    for p in range(81):
        if board[p]: continue
        res2=G.play(board,p,me)
        if res2 is None: continue
        nb,cap,_=res2
        g2,l2=G.group(nb,p)
        tcap=len(cap)>0
        tunsafe=(not tcap) and len(l2)<=1
        own=[q for q in G.NB[p] if board[q]==me]
        oppq=[q for q in G.NB[p] if board[q]==opp]
        tesc=any(tl[q]==1 for q in own)
        ne=sum(1 for n in G.NB[p] if board[n]==0)
        res['legal']+=1; res['tcap']+=tcap; res['tunsafe']+=tunsafe; res['tesc']+=tesc
        for R in (3,4,5):
            dcap=any(len(Cx[R][q])<=1 for q in oppq)
            res[('cap',R,'fp')]+=dcap and not tcap
        for R in (3,4):
            C=Co[R]
            other=set()
            for q in own: other|=C[q]
            other.discard(p3(p))
            for n in G.NB[p]:
                if board[n]==0: other.add(p3(n))
            dun1=not(len(other)>=2 or ne>=2)
            mp=max([len(C[q])-1 for q in own]+[0])
            dun2= ne+mp<=1
            for nm,du in (('merge',dun1),('class',dun2)):
                res[('uns',R,nm,'fp')]+=du and not tunsafe and not tcap
                res[('uns',R,nm,'fn')]+=(not du) and tunsafe
            desc=any(len(C[q]-{p3(p)})==0 for q in own)
            res[('esc',R,'fp')]+=desc and not tesc
            res[('esc',R,'fn')]+=tesc and not desc
for k in sorted(res,key=str): print(k,res[k])
```