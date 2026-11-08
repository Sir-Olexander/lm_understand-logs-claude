```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
def cats(passno,opp='strong'):
    d=json.load(open('/work/pass_%d/games.json'%passno))
    B=np.load('/work/pass_%d/boards.npy'%passno)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    cat=collections.Counter()
    for gi,g in enumerate(d):
        if g['opponent']!=opp: continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            prev=tob(B[gi,k-1]); p=m['move']
            r=G.play(prev,p,me)
            if r is None: continue
            nb,cap,_=r
            g2,l2=G.group(nb,p)
            if len(l2)==1 and not cap and len(g2)>=2:
                ne=sum(1 for n in G.NB[p] if prev[n]==0)
                gl=[];seen=set()
                for n in G.NB[p]:
                    if prev[n]==me:
                        gg,ll=G.group(prev,n)
                        if min(gg) in seen: continue
                        seen.add(min(gg)); gl.append(len(ll))
                eye=all(prev[n]==me for n in G.NB[p])
                cat[(ne,tuple(sorted(gl)),eye)]+=1
    return cat
print(cats(3).most_common(8))
print(cats(4).most_common(8))
```