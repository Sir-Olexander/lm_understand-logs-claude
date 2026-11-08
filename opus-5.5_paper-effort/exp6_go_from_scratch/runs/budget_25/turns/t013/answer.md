```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def tob(b):  # to engine board: 1 black, 2 white
    return [1 if x==1 else (2 if x==-1 else 0) for x in b]
stats=collections.Counter()
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    mycol=1 if g['net_color']=='black' else 2
    mv=g['moves']
    for k,m in enumerate(mv):
        if m['by']!='net' or m['move']==81: continue
        prev=tob(B[gi,k-1]) if k>0 else [0]*81
        gid,groups=G.all_groups(prev)
        # opportunities
        capmax=0; savemax=0
        for gg,l,c in groups:
            if len(l)==1:
                lib=next(iter(l))
                if c!=mycol: capmax=max(capmax,len(gg))
                else:
                    res=G.play(prev,lib,mycol)
                    if res:
                        g2,l2=G.group(res[0],lib)
                        if len(l2)>=2: savemax=max(savemax,len(gg))
        res=G.play(prev,m['move'],mycol)
        if res is None: stats['illegal?']+=1; continue
        nb,cap,_=res
        g2,l2=G.group(nb,m['move'])
        mycap=len(cap)
        if capmax>=1:
            stats['cap_opp_avail']+=1
            if mycap>0: stats['cap_taken']+=1
            if capmax>=2:
                stats['cap2_avail']+=1
                if mycap>=2: stats['cap2_taken']+=1
        if savemax>=2:
            stats['save2_avail']+=1
        if len(l2)==1 and mycap==0:
            stats['selfatari']+=1
            if len(g2)>=2: stats['selfatari2+']+=1
        stats['moves']+=1
print(stats)
```