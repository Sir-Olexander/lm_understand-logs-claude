```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np
d=json.load(open('/work/pass_19/games.json')); B=np.load('/work/pass_19/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
for gi,g in enumerate(d):
    if g['end']!='repetition' or g['result']!=-1: continue
    n=len(g['moves']); last=B[gi,n-1]; me=1 if g['net_color']=='black' else 2
    match=[k for k in range(n-2) if (B[gi,k]==last).all()][-1]
    caps=[int(((B[gi,k-1]!=0)&(B[gi,k]==0)).sum()) for k in range(match+1,n)]
    seq=[(m['by'][0],divmod(m['move'],9),round(dict(m.get('top5',[])).get(m['move'],0),2)) for m in g['moves'][match+1:]]
    # first net move in cycle: self-atari?
    ply=match+1 if g['moves'][match+1]['by']=='net' else match+2
    prev=tob(B[gi,ply-1]); mv=g['moves'][ply]['move']
    desc=''
    if mv!=81:
        r=G.play(prev,mv,me); g2,l2=G.group(r[0],mv); desc='libs %d size %d cap %d'%(len(l2),len(g2),len(r[1]))
    print(gi,g['margin'],'len',n-1-match,seq,caps,'| first net move',desc)
```