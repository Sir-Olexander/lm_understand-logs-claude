```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import deque
G=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def show(b,mark=None):
    return '\n'.join(''.join('*' if r*9+c==mark else '.XO'[b[r*9+c]] for c in range(9)) for r in range(9))
E=lambda b,i: sum(1 for j in go.NB[i] if b[j]==0)
def within(b,i,S,R=3):
    col=b[i]; dq=deque([(i,0)]); seen={i}
    while dq:
        x,dd=dq.popleft()
        if x in S: return True
        if dd==R: continue
        for y in go.NB[x]:
            if b[y]==col and y not in seen: seen.add(y); dq.append((y,dd+1))
    return False
def flags(b,c):
    e={i:E(b,i) for i in range(81) if b[i]==c}
    sig={i:sum(e[j] for j in go.NB[i] if b[j]==c) for i in e}
    Fn=set(i for i in e if 2.5*e[i]+sig[i]-2.5>0)
    F3n=set(i for i in e if 2*e[i]+sig[i]-4.5>0)
    return Fn,F3n
cats={}
for gi,g in enumerate(G):
    netcol = -1 if g['net_color']=='white' else 1
    me=1 if netcol==1 else 2; op=3-me
    mv=g['moves']
    for k in range(1,len(mv)-1):
        m=mv[k]
        if m['by']!='net' or m['move']==81: continue
        p=m['move']
        if B[gi,k+1][p]!=netcol:
            b=conv(B[gi,k-1])
            Fo,F3o=flags(b,op); Fm,F3m=flags(b,me)
            falsecap=[j for j in go.NB[p] if b[j]==op and not within(b,j,Fo|F3o) and len(go.group(b,j)[1])>=2]
            truecap=[j for j in go.NB[p] if b[j]==op and len(go.group(b,j)[1])==1]
            ownat=[j for j in go.NB[p] if b[j]==me and not within(b,j,Fm|F3m)]
            n3=[j for j in go.NB[p] if b[j]==me and within(b,j,F3m)]
            key=('falsecap' if falsecap else 'truecap' if truecap else 'esc' if ownat else 'other')
            cats.setdefault(key,[]).append((gi,k,E(b,p),len(n3)))
for kk,v in cats.items(): print(kk,len(v),v[:12])
```