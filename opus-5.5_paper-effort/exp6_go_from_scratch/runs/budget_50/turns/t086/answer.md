```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(18)
tot=0; nwith=0; ex=None
for i,g in enumerate(G):
    if not (-10<g['margin']<0): continue
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves']); bd=B[i,n-1]
    dead=[]
    for c,st,l in groups(bd):
        if c!=-me: continue
        nb_nonempty={int(bd[q]) for s_ in st for q in NBL[s_] if bd[q]!=0 and q not in st}
        if nb_nonempty<= {me} and len(l)<=2: dead.append(len(st))
    if dead: nwith+=1; tot+=sum(dead); ex=ex or (i,dead)
print('close losses with capturable dead opp stones at end:',nwith,'stones',tot)
i,d=ex; print(d, G[i]['net_color'], G[i]['margin']); print(show(B[i,len(G[i]['moves'])-1]))
```