```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
d=json.load(open('/work/pass_17/games.json')); B=np.load('/work/pass_17/boards.npy')
print(sorted((g['margin'],g['end'][:3],g['net_color'][0]) for g in d if g['result']==-1))
for gi,g in enumerate(d):
    if g['result']!=-1 or g['end']!='two passes' or g['margin']<-6: continue
    n=len(g['moves'])
    fin=tob(B[gi,n-1])
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
    lastnet=[m for m in g['moves'] if m['by']=='net'][-1]
    print(gi,g['net_color'][0],g['margin'],'dame',dame,'first passer',g['moves'][-2]['by'],'net final top5',[(divmod(a,9),p) for a,p in lastnet['top5'][:3]])
```