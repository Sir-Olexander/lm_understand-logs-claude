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
COL={'par2':par,'p3':p3,'mod3':mod3}
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
            if par(s)==1:
                if m>=1: st.add(0)
                if m>=2: st.add(1)
                if m>=3: st.add(2)
            else:
                nB=sum(1 for n in libs if p3(n)==1); nC=m-nB
                if 2*nB+nC>=2: st.add(1)
                if 2*nC+nB>=2: st.add(2)
                if m>=3: st.add(0)
        elif scheme=='mod3':
            c=mod3(s); npl=sum(1 for n in libs if mod3(n)==(c+1)%3); nmi=m-npl
            if 2*npl+nmi>=2: st.add((c+1)%3)
            if 2*nmi+npl>=2: st.add((c-1)%3)
            if m>=3: st.add(c)
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
for board,me in positions[::4]:
    gid,groups=G.all_groups(board)
    tl={s:len(groups[gid[s]][1]) for s in range(81) if board[s]}
    Cs={(sc,R):prop(board,me,seeds(board,me,sc),R) for sc in ('par2','p3','mod3') for R in (3,4,5,12)}
    for p in range(81):
        if board[p]: continue
        r2=G.play(board,p,me)
        if r2 is None: continue
        nb,cap,_=r2
        g2,l2=G.group(nb,p)
        tcap=len(cap)>0
        tuns=(not tcap) and len(l2)<=1
        own=[q for q in G.NB[p] if board[q]==me]
        tat=any(tl[q]==1 for q in own)
        ne=sum(1 for n in G.NB[p] if board[n]==0)
        res['legal']+=1; res['tuns']+=tuns; res['tat']+=tat
        for (sc,R),C in Cs.items():
            cf=COL[sc]
            dat=any(len(C[q])<=1 for q in own)
            res[(sc,R,'at_fp')]+=dat and not tat
            other=set()
            for q in own: other|=C[q]
            other.discard(cf(p))
            for n in G.NB[p]:
                if board[n]==0: other.add(cf(n))
            mu=not(len(other)>=2 or ne>=2)
            mp=max([len(C[q])-1 for q in own]+[0])
            cu= ne+mp<=1
            for nm,du in (('merge',mu),('class',cu),('both',mu and cu)):
                res[(sc,R,nm,'fp')]+=du and not tuns and not tcap
                res[(sc,R,nm,'fn')]+=(not du) and tuns
print('legal',res['legal'],'tuns',res['tuns'],'tat',res['tat'])
for sc in ('par2','p3','mod3'):
    for R in (3,4,5,12):
        print(sc,R,'at_fp',res[(sc,R,'at_fp')],' '.join('%s fp %d fn %d'%(nm,res[(sc,R,nm,'fp')],res[(sc,R,nm,'fn')]) for nm in ('merge','class','both')))
```