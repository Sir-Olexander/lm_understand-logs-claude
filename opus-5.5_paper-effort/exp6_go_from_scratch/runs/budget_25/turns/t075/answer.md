```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_10/games.json'))
B=np.load('/work/pass_10/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cat=collections.Counter(); probs=collections.defaultdict(list)
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
        if len(l2)==1 and not cap:
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            gl=[];seen=set()
            for n in G.NB[p]:
                if prev[n]==me:
                    gg,ll=G.group(prev,n)
                    if min(gg) in seen: continue
                    seen.add(min(gg)); gl.append(len(ll))
            oppbig=max([len(G.group(prev,n)[0]) for n in G.NB[p] if prev[n]==3-me]+[0])
            key=(ne,tuple(sorted(gl)),'big' if oppbig>=6 else 'small')
            cat[key]+=1
            pr=dict((a,b) for a,b in m['top5']).get(p,0); probs[key].append(pr)
for k,v in cat.most_common(12): print(k,v,'mean prob',round(np.mean(probs[k]),2))
```