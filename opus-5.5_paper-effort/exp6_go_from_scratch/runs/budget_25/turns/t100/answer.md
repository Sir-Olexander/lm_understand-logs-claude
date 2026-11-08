```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter()
for ps in (13,14,15):
    d=json.load(open('/work/pass_%d/games.json'%ps)); B=np.load('/work/pass_%d/boards.npy'%ps)
    for gi,g in enumerate(d):
        if g['opponent']!='strong': continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            prev=tob(B[gi,k-1]); p=m['move']
            r=G.play(prev,p,me)
            if not r or len(r[1])<6: continue
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            own=[n for n in G.NB[p] if prev[n]==me]
            kadj=sum(1 for n in G.NB[p] if prev[n]==3-me and len(G.group(prev,n)[1])==1)
            # own nbr min libs
            ol=min([len(G.group(prev,n)[1]) for n in own]+[9])
            cat[(len(own),ne,min(ol,3),kadj)]+=1
for k,v in cat.most_common(15): print('kown,ne,ownlibs,kadj',k,v)
print(sum(cat.values()))
```