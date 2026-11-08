```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(21)
def regions(bd):
    seen=set(); out=[]
    for p in range(81):
        if bd[p]!=0 or p in seen: continue
        st=[p]; seen.add(p); border=set(); k=0
        while k<len(st):
            q=st[k]; k+=1
            for m in NBL[q]:
                if bd[m]==0 and m not in seen: seen.add(m); st.append(m)
                elif bd[m]!=0: border.add(int(bd[m]))
        out.append((st,border))
    return out
for i,g in enumerate(G):
    if not(-4<g['margin']<0): continue
    bd=B[i,len(g['moves'])-1]
    neu=[ [divmod(p,9) for p in st] for st,bo in regions(bd) if len(bo)==2]
    print(g['net_color'][0],g['margin'],g['end'][:3],neu)
```