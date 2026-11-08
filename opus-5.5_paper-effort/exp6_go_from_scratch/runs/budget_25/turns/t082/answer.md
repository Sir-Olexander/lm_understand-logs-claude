```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,Us=pickle.load(open('/work/v8_units.pkl','rb'))
items=json.load(open('/work/pos11_items.json'))
A={l:np.load('/work/pass_11/act_L%d.npy'%l) for l in range(1,7)}
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
def U(key): return Us[str(tuple(key))]
errs={}
def err(k,m): errs.setdefault(k,[]).append(m)
SK=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44, 62,10,16,64])
for i,it in enumerate(items):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board'])
    board=[0 if v==0 else (1 if v==me else 2) for v in bd]
    Co=prop(board,1,seeds(board,1),4); Cx4=prop(board,2,seeds2(board,2),4); Cx5=prop(board,2,seeds2(board,2),5)
    n4c={}
    for q in range(81):
        if q not in SK:
            I1=sum((board[t]==1)-(board[t]==2) for t in G.NB[q])
            exp= max(0,I1) if (board[q]==2 and par(q) not in Cx4[q]) else 0
            n4c[q]=exp
            v=A[5][i,U(('n4c',5,q))]
            if abs(v-exp)>1e-3: err('n4c',(i,q,v,exp))
        exp=1.0 if (board[q]==2 and par(q) not in Cx5[q] and any(board[n]==0 for n in G.NB[q])) else 0.0
        v=A[6][i,U(('n5',6,q))]
        if board[q]==2 and not any(board[n]==0 for n in G.NB[q]): continue
        if abs(v-exp)>1e-3: err('n5',(i,q,v,exp))
    for p in range(81):
        if board[p]: continue
        ne=sum(1 for n in G.NB[p] if board[n]==0)
        own=[q for q in G.NB[p] if board[q]==1]
        W=sum(len(Co[q])-2 for q in own)
        sup=sum(n4c.get(q,0) for q in G.NB[p])
        uc=max(0,P['Ua']-P['Ub']*ne-P['Uw']*W-P['Uk']*len(own)-P['BIGC']*sup-0.3)
        v=A[6][i,U(('Uc',6,p))]
        if abs(v-uc)>0.05: err('Uc',(i,p,v,uc))
for k,v in errs.items(): print(k,len(v),v[:5])
print('ok')
```