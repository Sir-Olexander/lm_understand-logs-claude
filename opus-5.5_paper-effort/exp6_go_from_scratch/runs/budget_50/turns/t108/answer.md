```python
import sys; sys.path.insert(0,'/work'); from ana import *
import numpy as np
from collections import Counter
G,B=load(24); st=Counter()
for i,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    n=len(g['moves'])
    for k in range(1,n):
        prev=B[i,k-1]; bd=B[i,k]
        dead=[p for p in range(81) if prev[p]==me and bd[p]==0]
        if len(dead)<5: continue
        # look back at our moves within 12 plies, libs of the string containing dead[0] at that time
        for m in g['moves'][max(0,k-12):k]:
            if m['by']!='net': continue
            b0=board_before(B,i,m['ply'])
            if b0[dead[0]]!=me: continue
            gl=[(s2,l) for c,s2,l in groups(b0) if dead[0] in s2][0]
            nl=len(gl[1]); st[('libs',min(nl,5))]+=1
            st[('libs',min(nl,5),'played_lib')]+= m['move'] in gl[1]
            st[('libs',min(nl,5),'pass')]+= m['move']==81
print(sorted(st.items()))
```