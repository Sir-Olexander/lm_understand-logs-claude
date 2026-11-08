```python
import sys; sys.path.insert(0,'/work'); from ana import *
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
            if cnt in (3,5):
                print(g['net_color'],'ply',k,'lost',lost)
                print([(m['by'][0],m['move'],(m.get('top5') or [[0,0]])[0]) for m in g['moves'][k-13:k+1]])
                print(showmany([B[i,j] for j in range(k-13,k+1,2)]))
            break
```