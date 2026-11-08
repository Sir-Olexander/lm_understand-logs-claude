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
shown=0
for i,g in enumerate(G):
    if not(-4<g['margin']<0): continue
    me=1 if g['net_color']=='black' else -1
    bd=B[i,len(g['moves'])-1]
    res=[]
    for st,bo in regions(bd):
        if len(bo)==2 and len(st)<=2:
            for p in st:
                nb,cap=play(bd,p,me); l=[ll for c,s2,ll in groups(nb) if p in s2][0]
                res.append((divmod(p,9),len(l),cap))
    print(g['net_color'][0],g['margin'],res)
    if shown<2 and g['margin']==-0.5: print(show(bd*me)); shown+=1
```