```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
def capstats(ps):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    B=np.load('/work/pass_%d/boards.npy'%ps)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    st=collections.Counter()
    for gi,g in enumerate(d):
        if g['opponent']!='strong': continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or k==0: continue
            prev=tob(B[gi,k-1])
            gid,groups=G.all_groups(prev)
            best=0
            for gg,l,c in groups:
                if len(l)==1 and c!=me:
                    lib=next(iter(l))
                    if G.play(prev,lib,me) is not None: best=max(best,len(gg))
            if best:
                key='cap1' if best==1 else ('cap2-4' if best<=4 else 'cap5+')
                st[key+'_avail']+=1
                if m['move']!=81:
                    r=G.play(prev,m['move'],me)
                    if r and r[1]: st[key+'_took']+=1
                else: st[key+'_pass']+=1
            if m['move']==81:
                st['pass']+=1
                if g['moves'][k-1]['move']==81: st['pass_after_opp_pass']+=1
    return st
for ps in (10,12):
    print(ps,sorted(capstats(ps).items()))
```