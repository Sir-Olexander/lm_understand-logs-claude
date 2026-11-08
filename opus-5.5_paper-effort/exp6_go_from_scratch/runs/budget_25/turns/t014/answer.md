```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def tob(b): return [1 if x==1 else (2 if x==-1 else 0) for x in b]
c_mod3=lambda s:(s//9+s%9)%3
c_par=lambda s:(s//9+s%9)%2
def seeds(board,side,scheme):
    S={}
    for s in range(81):
        if board[s]!=side: continue
        libs=[n for n in G.NB[s] if board[n]==0]
        if scheme=='mod3':
            c=c_mod3(s); npl=sum(1 for n in libs if c_mod3(n)==(c+1)%3); nmi=len(libs)-npl
            st=set()
            if npl>=1 or nmi==2: st.add((c+1)%3)
            if nmi>=1 or npl==2: st.add((c-1)%3)
        else:
            st=set()
            if len(libs)>=1: st.add(1-c_par(s))
            if len(libs)>=2: st.add(c_par(s))
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
print(len(positions))
res=collections.Counter()
for board,me in positions[::2]:
    opp=3-me
    gid,groups=G.all_groups(board)
    truth_lib={s:len(groups[gid[s]][1]) for s in range(81) if board[s]}
    for scheme in ('par2','mod3'):
        S=seeds(board,opp,scheme)
        for R in (3,4):
            C=prop(board,opp,S,R)
            for p in range(81):
                if board[p]: continue
                qs=[q for q in G.NB[p] if board[q]==opp]
                if not qs: continue
                tcap=any(truth_lib[q]==1 for q in qs)
                dcap=any(len(C[q])<=1 for q in qs)
                key=(scheme,R)
                res[key+('tcap',)]+=tcap
                res[key+('fp',)]+=(dcap and not tcap)
                res[key+('fn',)]+=(tcap and not dcap)
for k in sorted(res): print(k,res[k])
```