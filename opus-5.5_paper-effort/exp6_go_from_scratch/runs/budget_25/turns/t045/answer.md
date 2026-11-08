```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_6/games.json'))
gs=[g for g in d if g['opponent']=='strong']
print(sorted((g['margin'],g['end'][:3]) for g in gs if g['result']==-1))
print(sum(1 for g in gs if g['margin']>60), 'wipeouts')
exec(open('/dev/stdin').read()) if False else None
def analyze(passno,opp='strong'):
    d=json.load(open('/work/pass_%d/games.json'%passno))
    B=np.load('/work/pass_%d/boards.npy'%passno)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    st=collections.Counter()
    for gi,g in enumerate(d):
        if g['opponent']!=opp: continue
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or k==0: continue
            prev=tob(B[gi,k-1])
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
                st['pass']+=1; continue
            r=G.play(prev,m['move'],me)
            if r is None: continue
            nb,cap,_=r
            g2,l2=G.group(nb,m['move'])
            if capmax>=1: st['cap_avail']+=1; st['cap_taken']+= len(cap)>0
            if savemax>=2: st['save2_avail']+=1; st['save2_done']+= (m['move'] in [next(iter(l)) for gg,l,c in groups if len(l)==1 and c==me and len(gg)>=2]) or len(cap)>0
            if len(l2)==1 and not cap:
                st['selfatari']+=1
                if len(g2)>=2: st['selfatari2+']+=1
    return st
print(analyze(6))
```