```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
st=collections.Counter()
for gi,g in enumerate(d):
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        prev=tob(B[gi,k-1]); p=m['move']
        nb,cap,_=G.play(prev,p,me)
        g2,l2=G.group(nb,p)
        alln=all(prev[n]==me for n in G.NB[p])
        if alln:
            st['eyefill']+=1
            if len(l2)<=1: st['eyefill_selfatari']+=1; st['eyefill_sa_size']+=len(g2)
        if len(l2)==1 and not cap:
            st['selfatari']+=1
            if len(g2)>=2: st['selfatari2+']+=1
print(st)
print(analyze if False else '')
```