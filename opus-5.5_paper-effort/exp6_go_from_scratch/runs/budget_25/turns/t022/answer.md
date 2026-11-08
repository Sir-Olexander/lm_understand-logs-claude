```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,Us=pickle.load(open('/work/v2_units.pkl','rb'))
items=json.load(open('/work/pos2_items.json'))
A={l:np.load('/work/pass_2/act_L%d.npy'%l) for l in range(1,7)}
X=np.load('/work/pass_2/inputs.npy')
par=lambda s:(s//9+s%9)%2
def p3(s):
    r,c=divmod(s,9)
    if (r+c)%2==0: return 0
    return 1 if r%2==0 else 2
def seeds(board,side,scheme):
    S={}
    for s in range(81):
        if board[s]!=side: continue
        libs=[n for n in G.NB[s] if board[n]==0]; m=len(libs); st=set()
        if scheme=='par2':
            if m>=1: st.add(1-par(s))
            if m>=2: st.add(par(s))
        else:
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
def U(key): return Us[str(key)]
errs={}
def err(k,msg):
    errs.setdefault(k,[]).append(msg)
for i,it in enumerate(items):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board'])
    board=[0 if v==0 else (1 if v==me else 2) for v in bd]  # 1=own,2=opp
    # check input
    for s in range(81):
        if X[i,s*3]!=(board[s]==1) or X[i,s*3+1]!=(board[s]==2): err('input',(i,s))
    Co=prop(board,1,seeds(board,1,'p3'),4)
    Cx=prop(board,2,seeds(board,2,'par2'),4)
    for s in range(81):
        bexp=1+(board[s]==1)-(board[s]==2)
        for l in range(1,6):
            if abs(A[l][i,U(('b',l,s))]-bexp)>1e-4: err('b%d'%l,(i,s))
        if abs(A[5][i,U(('O',5,s))]-(board[s]==1))>1e-4: err('O5',(i,s))
        ks=[1,2] if par(s)==1 else [0,1,2]
        for k in ks:
            exp= 1.0 if (board[s]==1 and k not in Co[s]) else 0.0
            if abs(A[5][i,U(('m',5,s,k))]-exp)>1e-4: err('m',(i,s,k,A[5][i,U(('m',5,s,k))],exp))
        exp=1.0 if (board[s]==2 and par(s) not in Cx[s]) else 0.0
        if abs(A[5][i,U(('n',5,s))]-exp)>1e-4: err('n',(i,s,A[5][i,U(('n',5,s))],exp))
        # L6 checks
        at_exp=1.0 if (board[s]==1 and len(Co[s])==1) else 0.0
        if board[s]==1 and any(board[n]==0 for n in G.NB[s]):
            if abs(A[6][i,U(('at',6,s))]-at_exp)>1e-4: err('at',(i,s))
        if board[s]==0:
            ne=sum(1 for n in G.NB[s] if board[n]==0)
            own=[q for q in G.NB[s] if board[q]==1]
            X2=sum(len(Co[q])-1 for q in own); W=sum(len(Co[q])-2 for q in own)
            u1=max(0,P['B1']*(1.5-ne)-P['BIG1']*X2); u0=max(0,P['B0']*(0.5-ne)-P['BIG0']*W)
            if abs(A[6][i,U(('U1',6,s))]-u1)>1e-3: err('U1',(i,s,A[6][i,U(('U1',6,s))],u1))
            if abs(A[6][i,U(('U0',6,s))]-u0)>1e-3: err('U0',(i,s,A[6][i,U(('U0',6,s))],u0))
            eye=1.0 if all(board[n]==1 for n in G.NB[s]) else 0.0
            if abs(A[6][i,U(('eye',6,s))]-eye)>1e-4: err('eye',(i,s))
for k,v in errs.items(): print(k,len(v),v[:5])
print('done; checked',len(items))
```