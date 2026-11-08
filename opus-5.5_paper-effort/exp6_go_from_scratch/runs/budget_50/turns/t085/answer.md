```python
import numpy as np, sys; sys.path.insert(0,'/work'); from ana import *
lg=np.load('/work/pass_19/logits.npy'); X=np.load('/work/pass_19/inputs.npy')
def area(bd):
    sc={1:0,-1:0}
    for p in range(81):
        if bd[p]: sc[int(bd[p])]+=1
    seen=set()
    for p in range(81):
        if bd[p] or p in seen: continue
        st=[p]; seen.add(p); bo=set(); k=0
        while k<len(st):
            q=st[k]; k+=1
            for m in NBL[q]:
                if bd[m]==0 and m not in seen: seen.add(m); st.append(m)
                elif bd[m]: bo.add(int(bd[m]))
        if len(bo)==1: sc[bo.pop()]+=len(st)
    return sc
for i in range(7):
    x=X[i].reshape(81,3); bd=(x[:,0]-x[:,1]).astype(int); white=x[0,2]
    sc=area(bd); true=sc[1]-sc[-1]+(7.5 if white else -7.5)
    print(i,'est',round((lg[i,81]+4)/0.4,1),'true(me)',true,'maxmove',round(float(lg[i,:81].max()),1))
```