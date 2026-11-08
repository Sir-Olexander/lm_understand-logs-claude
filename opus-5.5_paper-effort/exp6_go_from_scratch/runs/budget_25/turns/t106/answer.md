```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
for ps in (13,14,16):
    d=json.load(open('/work/pass_%d/games.json'%ps)); B=np.load('/work/pass_%d/boards.npy'%ps)
    for gi,g in enumerate(d):
        if g['opponent']!='strong' or g['result']!=-1 or g['end']!='two passes' or g['margin']<-6: continue
        n=len(g['moves']); me=1 if g['net_color']=='black' else 2
        fin=tob(B[gi,n-1])
        # dame: empty regions bordering both
        seen=set(); dame=0
        for s in range(81):
            if fin[s]==0 and s not in seen:
                st=[s]; reg={s}; bd=set()
                while st:
                    x=st.pop()
                    for nn in G.NB[x]:
                        if fin[nn]==0 and nn not in reg: reg.add(nn); st.append(nn)
                        elif fin[nn]: bd.add(fin[nn])
                seen|=reg
                if len(bd)==2: dame+=len(reg)
        # who passed first in final double pass
        firstpass=g['moves'][-2]['by']
        netpasses=sum(1 for m in g['moves'] if m['by']=='net' and m['move']==81)
        print(ps,gi,g['net_color'][0],g['margin'],'dame',dame,'first passer',firstpass,'net passes',netpasses,'len',n)
```