```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle, collections
P,Us=pickle.load(open('/work/v8_units.pkl','rb'))
items=json.load(open('/work/pos11_items.json'))
A5=np.load('/work/pass_11/act_L5.npy')
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
T=[];F=[]
for i,it in enumerate(items):
    me=1 if it['to_move']=='black' else 2
    bd=G.parse(it['board']); board=[0 if v==0 else (1 if v==me else 2) for v in bd]
    C4=prop(board,2,seeds2(board,2),4)
    for q in range(81):
        if board[q]!=2 or par(q) in C4[q] or not any(board[n]==0 for n in G.NB[q]): continue
        V=A5[i,Us[str(('v',5,q))]]
        gg,ll=G.group(board,q)
        (T if len(ll)==1 else F).append((V,len(gg)))
T=np.array(T); F=np.array(F)
print('true n',len(T),'FP n',len(F))
for th in (5,10,20,40,80,160):
    print(th,'true kept',(T[:,0]<th).mean().round(3),'FP kept',(F[:,0]<th).mean().round(3) if len(F) else None)
print('true V by size'); 
for sz in sorted(set(T[:,1])): print(int(sz), np.round(np.sort(T[T[:,1]==sz][:,0]),1)[:12])
print('FP V', np.round(np.sort(F[:,0]),1))
```