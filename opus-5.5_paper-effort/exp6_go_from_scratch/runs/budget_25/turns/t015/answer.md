```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def tob(b): return [1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter()
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        prev=tob(B[gi,k-1]) if k>0 else [0]*81
        p=m['move']
        res=G.play(prev,p,me)
        nb,cap,_=res
        g2,l2=G.group(nb,p)
        if len(l2)==1 and not cap and len(g2)>=2:
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            gl=[]
            seen=set()
            for n in G.NB[p]:
                if prev[n]==me:
                    gg,ll=G.group(prev,n)
                    if min(gg) in seen: continue
                    seen.add(min(gg)); gl.append(len(ll))
            cat[(ne,tuple(sorted(gl)))]+=1
for k,v in cat.most_common(20): print(k,v)
```