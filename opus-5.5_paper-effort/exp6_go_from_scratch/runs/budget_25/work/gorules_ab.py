
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
