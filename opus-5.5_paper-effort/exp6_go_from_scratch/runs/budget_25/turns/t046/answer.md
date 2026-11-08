```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter(); examples=collections.defaultdict(list)
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
        if len(l2)==1 and not cap and len(g2)>=2:
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            gl=[];seen=set()
            for n in G.NB[p]:
                if prev[n]==me:
                    gg,ll=G.group(prev,n)
                    if min(gg) in seen: continue
                    seen.add(min(gg)); gl.append(len(ll))
            # adjacent opp groups with 1 lib (true) or not
            oppl=sorted(len(G.group(prev,n)[1]) for n in G.NB[p] if prev[n]==3-me)
            key=(ne,tuple(sorted(gl)))
            cat[key]+=1; examples[key].append((gi,k,p,oppl,m['top5'][:2]))
print(cat.most_common(8))
for key in [(0,(2,)),(1,(1,))]:
    for gi,k,p,oppl,top in examples[key][:4]:
        print(key,gi,k,divmod(p,9),'opp nbr libs',oppl,[(divmod(a,9),pr) for a,pr in top])
        print(G.show(tob(B[gi,k-1])))
```