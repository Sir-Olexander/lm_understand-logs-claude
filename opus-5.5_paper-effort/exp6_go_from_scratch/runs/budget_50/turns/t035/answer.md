```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
import numpy as np
cnt=0
for i,g in enumerate(G):
    if g['opponent']!='strong' or g['result']>0: continue
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves']); prev=np.zeros(81,np.int8)
    for k in range(n):
        bd=B[i,k]; lost=int(((prev==me)&(bd==0)).sum()); prev=bd
        if lost>=5:
            cnt+=1
            if cnt in (2,4):
                print(g['net_color'],'ply',k,'lost',lost,[(m['by'][0],m['move'],(m.get('top5') or [[0,0]])[0]) for m in g['moves'][k-7:k+1]])
                print(showmany([B[i,j] for j in range(k-7,k+1)]))
            break
```