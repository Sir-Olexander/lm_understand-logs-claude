```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(5)
import numpy as np
res=[]
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves']); prev=np.zeros(81,np.int8); big_l=0; big_c=0
    for k in range(n):
        bd=B[i,k]; lost=int(((prev==me)&(bd==0)).sum()); capd=int(((prev==-me)&(bd==0)).sum())
        if lost>=4: big_l+=lost
        if capd>=4: big_c+=capd
        prev=bd
    fin=B[i,n-1]
    res.append((g['result'],g['margin'],big_l,big_c,int((fin==me).sum()),int((fin==-me).sum()),g['end']))
for r in sorted(res,key=lambda x:x[1])[:40:2]: print(r)
```