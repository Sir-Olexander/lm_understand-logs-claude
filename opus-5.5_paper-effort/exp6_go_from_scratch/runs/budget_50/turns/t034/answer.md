```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
import numpy as np
res=[]
for i,g in enumerate(G):
    if g['opponent']!='strong' or g['result']>0: continue
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves']); prev=np.zeros(81,np.int8); big_l=0; big_c=0
    for k in range(n):
        bd=B[i,k]; lost=int(((prev==me)&(bd==0)).sum()); capd=int(((prev==-me)&(bd==0)).sum())
        if lost>=3: big_l+=lost
        if capd>=3: big_c+=capd
        prev=bd
    res.append((g['net_color'][0],g['margin'],big_l,big_c,g['end'][:3]))
print(sorted(res,key=lambda x:x[1]))
```