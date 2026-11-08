```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,Us=pickle.load(open('/work/v5_units.pkl','rb'))
items=json.load(open('/work/pos7_items.json'))
A={l:np.load('/work/pass_7/act_L%d.npy'%l) for l in range(1,7)}
LG=np.load('/work/pass_7/logits.npy')
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
def U(key): return Us[str(tuple(key)) if isinstance(key,tuple) else str(key)]
SK=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44, 62,10])
errs={}
def err(k,m): errs.setdefault(k,[]).append(m)
for i,it in enumerate(items):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board'])
    board=[0 if v==0 else (1 if v==me else 2) for v in bd]
    Co=prop(board,1,seeds(board,1),4); Cx4=prop(board,2,seeds2(board,2),4)
    nO=sum(1 for v in board if v==1); nP=sum(1 for v in board if v==2)
    if abs(A[6][i,Us["('gS', 6)"]]-max(0,nO+nP-P['S0']))>1e-3: err('gS',i)
    if abs(A[6][i,Us["('gD', 6)"]]-max(0,nO-nP-P['D0']))>1e-3: err('gD',i)
    for s in range(81):
        if board[s]: continue
        ne=sum(1 for n in G.NB[s] if board[n]==0)
        own=[q for q in G.NB[s] if board[q]==1]
        W=sum(len(Co[q])-2 for q in own)
        n4=sum(1 for q in G.NB[s] if board[q]==2 and par(q) not in Cx4[q])
        eyeterm=P['EYEU']*(len(own)-len(G.NB[s])+1)
        u0=max(0,P['B0']*(0.5-ne)-P['BIG0']*W-P['BIGC']*n4+eyeterm-1.0)
        if abs(A[6][i,U(('U0',6,s))]-u0)>0.4: err('U0',(i,s,A[6][i,U(('U0',6,s))],u0))
        r,c=divmod(s,9); Ao=Ax=0.0
        for t in range(81):
            dd=abs(t//9-r)+abs(t%9-c)
            if dd==0 or dd>3: continue
            w=P['kern'][dd]
            Ao+=w*(board[t]==1)
            if t not in SK: Ax+=w*(board[t]==2)
        p1=P['C']+(P['lam']-P['mu'])*Ax-P['mu']*Ao; p2=max(0,Ax-Ao)
        if abs(A[6][i,U(('P1',6,s))]-p1)>0.01: err('P1',(i,s))
        if abs(A[6][i,U(('P2',6,s))]-p2)>0.01: err('P2',(i,s,A[6][i,U(('P2',6,s))],p2))
for k,v in errs.items(): print(k,len(v),v[:4])
print('ok')
```