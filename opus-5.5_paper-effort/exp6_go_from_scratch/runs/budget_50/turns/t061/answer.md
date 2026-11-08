```python
import sys,json; sys.path.insert(0,'/work'); from ana import *
G,B=load(12)
import numpy as np
cnt=0
for i,g in enumerate(G):
    if g['result']>0: continue
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves']); prev=np.zeros(81,np.int8)
    for k in range(n):
        bd=B[i,k]; lost=int(((prev==me)&(bd==0)).sum()); prev=bd
        if lost>=6:
            cnt+=1
            if cnt==3:
                mm=[m['move'] if m['move']!=81 else 'pass' for m in g['moves']]
                print(json.dumps(mm[:40])); print(json.dumps(mm[:42])); print(show(B[i,39]))
            break
```