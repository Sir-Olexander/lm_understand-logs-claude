```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
shown=0
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81 or k==0: continue
        prev=tob(B[gi,k-1]); p=m['move']
        r=G.play(prev,p,me)
        if r is None: continue
        nb,cap,_=r
        g2,l2=G.group(nb,p)
        ne=sum(1 for n in G.NB[p] if prev[n]==0)
        if len(l2)==1 and not cap and len(g2)>=2 and ne==0:
            if shown<5:
                print(gi,k,'me',me,'move',divmod(p,9),'size',len(g2),[(divmod(a,9),pr) for a,pr in m['top5']])
                print(G.show(prev)); shown+=1
```