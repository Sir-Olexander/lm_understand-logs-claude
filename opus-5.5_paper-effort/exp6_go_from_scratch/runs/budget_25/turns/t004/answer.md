```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import random
def heur_move(board,col,ko,rng):
    legal=[s for s in G.legal_moves(board,col,ko) if not all(board[n]==col for n in G.NB[s])]
    if not legal: return 81
    caps=[s for s in legal if G.play(board,s,col,ko)[1]]
    if caps and rng.random()<0.7: return rng.choice(caps)
    return rng.choice(legal)
def gen_games(n,seed=0):
    rng=random.Random(seed); pos=[]
    for gi in range(n):
        b=[0]*81; col=1; ko=-1; passes=0
        for t in range(162):
            m=heur_move(b,col,ko,rng)
            if m==81:
                passes+=1
                if passes==2: break
                ko=-1
            else:
                passes=0
                b,cap,ko=G.play(b,m,col,ko)
            col=3-col
            pos.append((list(b),col))
    return pos
pos=gen_games(40)
def eval_coloring(pos, colorf, K, R=4, cert_pair=False):
    st={'atari':0,'fp1':0,'n2':0,'fp2':0,'fpsize':[]}
    for board,col in pos[::4]:
        gid,groups=G.all_groups(board)
        seed={}
        for s in range(81):
            if board[s]==0: continue
            libs=[n for n in G.NB[s] if board[n]==0]
            cs=set(colorf(n) for n in libs)
            if len(libs)>=2: cs=set(range(K))
            seed[s]=cs
        if cert_pair:
            for s in range(81):
                if board[s]==0: continue
                for n in G.NB[s]:
                    if board[n]==board[s]:
                        u=set(x for x in G.NB[s]+G.NB[n] if board[x]==0)
                        if len(u)>=2: seed[s]=set(range(K))
        cur=seed
        for _ in range(R):
            new={}
            for s in cur:
                S=set(cur[s])
                for n in G.NB[s]:
                    if board[n]==board[s]: S|=cur[n]
                new[s]=S
            cur=new
        for s in range(81):
            if board[s]==0 or not any(board[n]==0 for n in G.NB[s]): continue
            g,l,c=groups[gid[s]]
            nl=len(l); nc=len(cur[s])
            if nl==1: st['atari']+=1
            elif nc<=1:
                st['fp1']+=1; st['fpsize'].append(len(g))
                if nl==2: st['fp2']+=1
            if nl==2: st['n2']+=1
    import numpy as np
    fs=st.pop('fpsize')
    st['fp_meansize']=round(float(np.mean(fs)),1) if fs else 0
    return st
cols={'mod5':(lambda s:(s//9+2*(s%9))%5,5),'mod3_rc':(lambda s:(s//9+s%9)%3,3),'par2':(lambda s:(s//9+s%9)%2,2),'mod4_rc2':(lambda s:((s//9)%2)*2+(s%9)%2,4),'one':(lambda s:0,1)}
for name,(f,K) in cols.items():
    for R in (3,4,5,8,20):
        print(name,R,eval_coloring(pos,f,K,R,True))
# group size distribution of atari groups
import collections
sz=collections.Counter()
for board,col in pos[::4]:
    gid,groups=G.all_groups(board)
    for g,l,c in groups:
        if len(l)==1: sz[min(len(g),10)]+=1
print(sorted(sz.items()))
```