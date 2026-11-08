```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,_=pickle.load(open('/work/v6_units.pkl','rb'))
D=json.load(open('/work/pos10.json')); items=D['items']; meta=D['meta']
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
SK=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44, 62,10])
def line(s):
    r,c=divmod(s,9); return min(r,c,8-r,8-c)+1
def evalp(board,p):
    Co=prop(board,1,seeds(board,1),4); Cx4=prop(board,2,seeds2(board,2),4); Cx5=prop(board,2,seeds2(board,2),5)
    ne=sum(1 for n in G.NB[p] if board[n]==0)
    own=[q for q in G.NB[p] if board[q]==1]; opp=[q for q in G.NB[p] if board[q]==2]
    cap=sum(1 for q in opp if par(q) not in Cx5[q])
    n4=sum(1 for q in opp if par(q) not in Cx4[q])
    esc=sum(1 for q in own if len(Co[q])==1)
    X2=sum(len(Co[q])-1 for q in own); W=sum(len(Co[q])-2 for q in own)
    u1=max(0,P['B1']*(2-ne)-P['BIG1']*X2-P['BIGC']*n4-0.2)
    eyet=P['EYEU']*(len(own)-len(G.NB[p])+1)
    u0=max(0,P['B0']*(0.5-ne)-P['BIG0']*W-P['BIGC']*n4+eyet-1)
    r,c=divmod(p,9); Ao=Ax=0
    for t in range(81):
        dd=abs(t//9-r)+abs(t%9-c)
        if dd==0 or dd>3: continue
        Ao+=P['kern'][dd]*(board[t]==1)
        if t not in SK: Ax+=P['kern'][dd]*(board[t]==2)
    S=sum(1 for v in board if v)
    pos=P['lam']*min(Ao,Ax)-P['mu']*(Ao+Ax)+P['eta']*ne+P['LV'][line(p)]+P['kL'].get(line(p),0)*max(0,S-50)
    tot=P['CAP']*cap+P['ESC']*esc-P['SA1']*u1-P['SA0']*u0+pos
    return dict(cap=cap,n4=n4,esc=esc,u1=round(u1,1),u0=round(u0,1),pos=round(pos,1),tot=round(tot,1))
for it,(gi,ply,mv) in zip(items,meta):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board']); board=[0 if v==0 else (1 if v==me else 2) for v in bd]
    vals=[(evalp(board,p)['tot'],p) for p in range(81) if board[p]==0]
    vals.sort(reverse=True)
    print(gi,ply,'played',divmod(mv,9),evalp(board,mv) if mv<81 else 'pass','| best',[(divmod(p,9),v) for v,p in vals[:3]])
```