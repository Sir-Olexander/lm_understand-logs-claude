```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
from collections import deque, Counter
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
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
R2={i:[j for j in range(81) if 0<abs(i//9-j//9)+abs(i%9-j%9)<=2] for i in range(81)}
st=Counter(); sizes=Counter()
for k in (8,9):
    G=json.load(open(f'/work/pass_{k}/games.json')); B=np.load(f'/work/pass_{k}/boards.npy')
    for gi,g in enumerate(G):
        me=1 if g['net_color']=='black' else 2; op=3-me
        for kk in range(1,len(g['moves'])):
            if g['moves'][kk]['by']!='net': continue
            b=conv(B[gi,kk-1])
            Fo,F3o=flags(b,op); Fm,F3m=flags(b,me)
            # candidate points: empty p adjacent to predicted-atari opp stone, e<=1, no n3 own
            for p in range(81):
                if b[p]: continue
                if E(b,p)>1: continue
                nb=[j for j in go.NB[p] if b[j]==op and not within(b,j,Fo|F3o)]
                if not nb: continue
                if any(b[j]==me and within(b,j,F3m) for j in go.NB[p]): continue
                true=[j for j in nb if len(go.group(b,j)[1])==1]
                dens=max(sum(1 for t in R2[j] if b[t]==op) for j in nb)
                if true:
                    sz=max(len(go.group(b,j)[0]) for j in true)
                    st[('true',min(dens,6))]+=1; sizes[min(sz,10)]+=1
                else:
                    st[('false',min(dens,6))]+=1
print('true by dens',[st[('true',d)] for d in range(7)])
print('false by dens',[st[('false',d)] for d in range(7)])
print(sorted(sizes.items()))
```