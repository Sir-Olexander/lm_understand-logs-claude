```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
def lossanalysis(ps):
    d=json.load(open('/work/pass_%d/games.json'%ps))
    B=np.load('/work/pass_%d/boards.npy'%ps)
    tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
    out=[]
    for gi,g in enumerate(d):
        if g['opponent']!='strong' or g['result']!=-1 or g['end']!='two passes': continue
        me=1 if g['net_color']=='black' else 2
        st=collections.Counter()
        for k,m in enumerate(g['moves']):
            if k==0: continue
            prev=tob(B[gi,k-1])
            if m['by']=='opponent':
                if m['move']!=81:
                    r=G.play(prev,m['move'],3-me)
                    if r and r[1]: st['lost_stones']+=len(r[1]); st['lost_events']+=1; st['maxlost']=max(st['maxlost'],len(r[1]))
                continue
            gid,groups=G.all_groups(prev)
            sv=0
            for gg,l,c in groups:
                if len(l)==1 and c==me:
                    lib=next(iter(l)); r=G.play(prev,lib,me)
                    if r and len(G.group(r[0],lib)[1])>=2: sv=max(sv,len(gg))
            if m['move']!=81:
                r=G.play(prev,m['move'],me)
                if r:
                    g2,l2=G.group(r[0],m['move'])
                    if len(l2)==1 and not r[1]: st['selfatari']+=1; st['sa_size']+=len(g2)
                    if sv>=3 and not r[1] and len(l2)<2: pass
                    if sv>=2: st['save_avail']+=1; st['saved']+= len(l2)>=2 and any(prev[n]==me for n in G.NB[m['move']])
        out.append((g['margin'],g['net_color'][0],dict(st)))
    return out
for x in lossanalysis(13)+lossanalysis(14): print(x)
```