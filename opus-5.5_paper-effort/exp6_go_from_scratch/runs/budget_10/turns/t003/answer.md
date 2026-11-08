```python
code = r'''
import numpy as np
N=9
NB=[[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        i=r*9+c
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: NB[i].append(rr*9+cc)
DIAG=[[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: DIAG[r*9+c].append(rr*9+cc)

def group(b,i):
    col=b[i]; st=[i]; seen={i}; libs=set()
    while st:
        x=st.pop()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in seen:
                seen.add(y); st.append(y)
    return seen,libs

class Game:
    def __init__(self):
        self.b=np.zeros(81,dtype=np.int8); self.turn=1; self.ko=-1; self.passes=0; self.n=0
        self.hist=[]
    def legal(self,i):
        if i==81: return True
        b=self.b
        if b[i]!=0 or i==self.ko: return False
        col=self.turn; opp=3-col
        for y in NB[i]:
            if b[y]==0: return True
        b[i]=col
        ok=False
        for y in NB[i]:
            if b[y]==opp:
                g,l=group(b,y)
                if len(l)==0: ok=True;break
            elif b[y]==col:
                g,l=group(b,y)
                if len(l)>0: ok=True;break
        b[i]=0
        return ok
    def legal_moves(self):
        return [i for i in range(81) if self.legal(i)]
    def play(self,i):
        self.n+=1
        if i==81:
            self.passes+=1; self.turn=3-self.turn; self.ko=-1; return []
        self.passes=0
        b=self.b; col=self.turn; opp=3-col
        b[i]=col; cap=[]
        for y in NB[i]:
            if b[y]==opp:
                g,l=group(b,y)
                if len(l)==0:
                    for z in g: b[z]=0
                    cap+=list(g)
        self.ko=-1
        if len(cap)==1:
            g,l=group(b,i)
            if len(g)==1 and len(l)==1: self.ko=cap[0]
        self.turn=opp
        return cap

def score(b):
    # tromp-taylor area: black, white
    s=[0,0,0]
    seen=set()
    for i in range(81):
        if b[i]: s[b[i]]+=1
        elif i not in seen:
            reg={i};st=[i];border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y);st.append(y)
                    elif b[y]: border.add(b[y])
            seen|=reg
            if border=={1}: s[1]+=len(reg)
            elif border=={2}: s[2]+=len(reg)
    return s[1],s[2]

def show(b):
    ch={0:'.',1:'X',2:'O'}
    return '\n'.join(''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9))

def is_own_eye(b,i,col):
    return b[i]==0 and all(b[y]==col for y in NB[i])
'''
open('/work/go.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, go; importlib.reload(go)
g=go.Game()
import random
random.seed(0)
for k in range(60):
    lm=[m for m in g.legal_moves() if not go.is_own_eye(g.b,m,g.turn)]
    g.play(random.choice(lm) if lm else 81)
print(go.show(g.b)); print(go.score(g.b))
```