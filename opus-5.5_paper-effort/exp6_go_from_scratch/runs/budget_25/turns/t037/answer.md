```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
def stats(passno,opp=None):
    d=json.load(open('/work/pass_%d/games.json'%passno))
    B=np.load('/work/pass_%d/boards.npy'%passno)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    st=collections.Counter()
    for gi,g in enumerate(d):
        if opp and g['opponent']!=opp: continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            prev=tob(B[gi,k-1]); p=m['move']
            r=G.play(prev,p,me)
            if r is None: st['none']+=1; continue
            nb,cap,_=r
            g2,l2=G.group(nb,p)
            if all(prev[n]==me for n in G.NB[p]):
                st['eyefill']+=1
                if len(l2)<=1: st['eyefill_selfatari']+=1; st['eyefill_sa_size']+=len(g2)
            if len(l2)==1 and not cap:
                st['selfatari']+=1
                if len(g2)>=2: st['selfatari2+']+=1
    return st
print(stats(3,'strong'))
print(stats(4,'strong'))
```