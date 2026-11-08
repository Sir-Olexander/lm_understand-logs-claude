```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,Us=pickle.load(open('/work/v4_units.pkl','rb'))
items=json.load(open('/work/pos2_items.json'))
A={l:np.load('/work/pass_5/act_L%d.npy'%l) for l in range(1,7)}
LG=np.load('/work/pass_5/logits.npy')
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
def U(key): return Us[str(tuple(key))]
errs={}
def err(k,msg): errs.setdefault(k,[]).append(msg)
own_colors=lambda q:[1,2] if par(q)==1 else [0,1,2]
for i,it in enumerate(items):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board'])
    board=[0 if v==0 else (1 if v==me else 2) for v in bd]
    Co=prop(board,1,seeds(board,1,'p3'),4)
    Cx4=prop(board,2,seeds(board,2,'par2'),4)
    Cx5=prop(board,2,seeds(board,2,'par2'),5)
    for s in range(81):
        ne=sum(1 for n in G.NB[s] if board[n]==0)
        O5=A[5][i,U(('O',5,s))]
        expO= 1.0 if board[s]==1 else (P['eps']*(ne+1) if board[s]==0 else 0.0)
        if abs(O5-expO)>2e-4 and board[s]!=0: err('O5',(i,s,O5,expO))
        if board[s]==0 and abs(O5/P['eps']-(ne+1))>0.05: err('O5z',(i,s,O5/P['eps'],ne+1))
        for k in own_colors(s):
            exp= 1.0 if (board[s]==1 and k not in Co[s]) else 0.0
            if abs(A[5][i,U(('m',5,s,k))]-exp)>2e-3: err('m',(i,s,k,A[5][i,U(('m',5,s,k))],exp))
        exp=1.0 if (board[s]==2 and par(s) not in Cx4[s]) else 0.0
        if abs(A[5][i,U(('n4',5,s))]-exp)>1e-4: err('n4',(i,s))
        exp=1.0 if (board[s]==2 and par(s) not in Cx5[s]) else 0.0
        if abs(A[6][i,U(('n5',6,s))]-exp)>1e-4: err('n5',(i,s,A[6][i,U(('n5',6,s))],exp))
        if board[s]==1 and ne>0:
            at_exp=1.0 if len(Co[s])==1 else 0.0
            if abs(A[6][i,U(('at',6,s))]-at_exp)>2e-3: err('at',(i,s))
        if board[s]==0:
            own=[q for q in G.NB[s] if board[q]==1]
            X2=sum(len(Co[q])-1 for q in own); W=sum(len(Co[q])-2 for q in own)
            n4=sum(1 for q in G.NB[s] if board[q]==2 and par(q) not in Cx4[q])
            u1=max(0,P['B1']*(2-ne)-P['BIG1']*X2-P['BIGC']*n4); u0=max(0,P['B0']*(0.5-ne)-P['BIG0']*W-P['BIGC']*n4)
            if abs(A[6][i,U(('U1',6,s))]-u1)>0.05: err('U1',(i,s,A[6][i,U(('U1',6,s))],u1))
            if abs(A[6][i,U(('U0',6,s))]-u0)>0.05: err('U0',(i,s,A[6][i,U(('U0',6,s))],u0))
            eye=1.0 if all(board[n]==1 for n in G.NB[s]) else 0.0
            if abs(A[6][i,U(('eye',6,s))]-eye)>2e-3: err('eye',(i,s))
            # pos
            r,c=divmod(s,9); pv=P['C']
            for t in range(81):
                dd=abs(t//9-r)+abs(t%9-c)
                if dd==0 or dd>3: continue
                pv+=P['prox_own'].get(dd,0)*(board[t]==1)+P['prox_opp'].get(dd,0)*(board[t]==2)*(1 if t not in {0,8,72,80,2,4,6,18,36,54,74,76,78,26,44} else 0)
            if abs(A[6][i,U(('pos',6,s))]-pv)>0.01: err('pos',(i,s,A[6][i,U(('pos',6,s))],pv))
for k,v in errs.items(): print(k,len(v),v[:4])
print('checked')
```