```python
code = r'''
import numpy as np, random
N=9
NB=[[] for _ in range(81)]
DG=[[] for _ in range(81)]
for r in range(N):
    for c in range(N):
        s=r*N+c
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<N and 0<=cc<N: NB[s].append(rr*N+cc)
        for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<N and 0<=cc<N: DG[s].append(rr*N+cc)

def group(board,s):
    col=board[s]; st=[s]; seen={s}; libs=set()
    while st:
        x=st.pop()
        for n in NB[x]:
            if board[n]==0: libs.add(n)
            elif board[n]==col and n not in seen:
                seen.add(n); st.append(n)
    return seen,libs

def all_groups(board):
    gid=[-1]*81; groups=[]
    for s in range(81):
        if board[s]!=0 and gid[s]<0:
            g,l=group(board,s)
            for x in g: gid[x]=len(groups)
            groups.append((g,l,board[s]))
    return gid,groups

def play(board,s,col,ko=-1):
    if s==81: return list(board),[],-1
    if board[s]!=0 or s==ko: return None
    b=list(board); b[s]=col; opp=3-col; cap=[]
    for n in NB[s]:
        if b[n]==opp:
            g,l=group(b,n)
            if not l:
                for x in g: b[x]=0
                cap.extend(g)
    g,l=group(b,s)
    if not l: return None
    newko=-1
    if len(cap)==1 and len(g)==1 and len(l)==1: newko=cap[0]
    return b,cap,newko

def legal_moves(board,col,ko=-1):
    return [s for s in range(81) if play(board,s,col,ko) is not None]

def area_score(board):
    sc={1:0,2:0}
    seen=set()
    for s in range(81):
        if board[s]: sc[board[s]]+=1
        elif s not in seen:
            st=[s]; reg={s}; border=set()
            while st:
                x=st.pop()
                for n in NB[x]:
                    if board[n]==0 and n not in reg: reg.add(n); st.append(n)
                    elif board[n]: border.add(board[n])
            seen|=reg
            if len(border)==1: sc[border.pop()]+=len(reg)
    return sc[1], sc[2]+7.5

def show(board):
    ch={0:'.',1:'X',2:'O'}
    return '\n'.join(''.join(ch[board[r*9+c]] for c in range(9)) for r in range(9))

def parse(rows):
    m={'.':0,'X':1,'O':2}
    return [m[ch] for row in rows for ch in row]
'''
open('/work/gorules_ab.py','w').write(code)
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
import time; t0=time.time()
pos=gen_games(40)
print(len(pos), time.time()-t0)

def eval_coloring(pos, colorf, K, R=4, cert_single=True, cert_pair=False):
    st={'atari':0,'fp1':0,'n2':0,'fp2':0,'n3':0,'j3':0}
    for board,col in pos[::4]:
        gid,groups=G.all_groups(board)
        seed={}
        for s in range(81):
            if board[s]==0: continue
            libs=[n for n in G.NB[s] if board[n]==0]
            cs=set(colorf(n) for n in libs)
            if cert_single and len(libs)>=2: cs=set(range(K))
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
                st['fp1']+=1
                if nl==2: st['fp2']+=1
            if nl==2: st['n2']+=1
            if nl>=3:
                st['n3']+=1
                if nc>=3: st['j3']+=1
    return st
colorings={
 'mod5_r2c':(lambda s:(s//9+2*(s%9))%5,5),
 'mod4_rc2':(lambda s:((s//9)%2)*2+(s%9)%2,4),
 'mod4_r2c':(lambda s:(s//9+2*(s%9))%4,4),
 'mod3_rc':(lambda s:(s//9+s%9)%3,3),
 'mod3_r2c':(lambda s:(s//9+2*(s%9))%3,3),
 'par2':(lambda s:(s//9+s%9)%2,2),
 'mod2_c':(lambda s:(s%9)%2,2),
 'mod2_floor':(lambda s:((s//9)//2+(s%9)//2)%2,2),
}
for name,(f,K) in colorings.items():
    for cp in (False,True):
        st=eval_coloring(pos,f,K,4,True,cp)
        print(name,cp,st, 'fp/multi=%.3f fp2/n2=%.3f j3/n3=%.3f'%(st['fp1']/(st['n2']+st['n3']), st['fp2']/st['n2'], st['j3']/st['n3']))
print(time.time()-t0)
```