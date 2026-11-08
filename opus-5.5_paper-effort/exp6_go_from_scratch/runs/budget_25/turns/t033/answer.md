```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
st=collections.Counter()
for passno in (1,3):
    d=json.load(open('/work/pass_%d/games.json'%passno))
    B=np.load('/work/pass_%d/boards.npy'%passno)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    for gi,g in enumerate(d):
        if g['opponent']!='strong': continue
        oc=2 if g['net_color']=='black' else 1
        for k,m in enumerate(g['moves']):
            if m['by']!='opponent' or m['move']==81: continue
            prev=tob(B[gi,k-1]) if k>0 else [0]*81
            p=m['move']
            nb,cap,_=G.play(prev,p,oc)
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            g2,l2=G.group(nb,p)
            st['moves']+=1
            if ne==0 and not cap:
                st['ne0']+=1
                if len(l2)<=1: st['ne0_selfatari']+=1
                # was it adjacent to own atari group (escape/connection)?
                own_at=any(prev[n]==oc and len(G.group(prev,n)[1])==1 for n in G.NB[p])
                if own_at: st['ne0_saveatari']+=1
                nown=len(set(min(G.group(prev,n)[0]) for n in G.NB[p] if prev[n]==oc))
                if nown>=2: st['ne0_connect2+']+=1
            if ne==1 and not cap: st['ne1']+=1
print(st)
```