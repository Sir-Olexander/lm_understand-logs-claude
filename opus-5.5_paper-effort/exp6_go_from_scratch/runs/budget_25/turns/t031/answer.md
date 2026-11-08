```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
def analyze(passno,opp='strong'):
    d=json.load(open('/work/pass_%d/games.json'%passno))
    B=np.load('/work/pass_%d/boards.npy'%passno)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    st=collections.Counter()
    for gi,g in enumerate(d):
        if g['opponent']!=opp: continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net': continue
            prev=tob(B[gi,k-1]) if k>0 else [0]*81
            gid,groups=G.all_groups(prev)
            capmax=0; savemax=0
            for gg,l,c in groups:
                if len(l)==1:
                    lib=next(iter(l))
                    if c!=me: capmax=max(capmax,len(gg))
                    else:
                        r=G.play(prev,lib,me)
                        if r:
                            g2,l2=G.group(r[0],lib)
                            if len(l2)>=2: savemax=max(savemax,len(gg))
            st['moves']+=1
            if m['move']==81:
                st['pass']+=1
                if capmax: st['pass_with_cap']+=1
                continue
            nb,cap,_=G.play(prev,m['move'],me)
            g2,l2=G.group(nb,m['move'])
            if capmax>=1:
                st['cap_avail']+=1; st['cap_taken']+= len(cap)>0
                if capmax>=2: st['cap2_avail']+=1; st['cap2_taken']+=len(cap)>=2
            if savemax>=1:
                st['save_avail']+=1
                # did we save the biggest?
            if savemax>=2: st['save2_avail']+=1
            if len(l2)==1 and not cap:
                st['selfatari']+=1
                if len(g2)>=2: st['selfatari2+']+=1
    return st
print(analyze(3))
print(analyze(1))
```