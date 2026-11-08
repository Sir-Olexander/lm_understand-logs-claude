```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def sa(ps):
    d=json.load(open('/work/pass_%d/games.json'%ps)); B=np.load('/work/pass_%d/boards.npy'%ps)
    st=collections.Counter(); ng=0
    for gi,g in enumerate(d):
        if g['opponent']!='strong': continue
        ng+=1
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if k==0: continue
            prev=tob(B[gi,k-1])
            if m['by']=='opponent':
                if m['move']!=81:
                    r=G.play(prev,m['move'],3-me)
                    if r and r[1]: st['lost']+=len(r[1])
                continue
            if m['move']==81: continue
            r=G.play(prev,m['move'],me)
            if r is None: continue
            g2,l2=G.group(r[0],m['move'])
            if r[1]: st['captured']+=len(r[1]); st['capbig']+= len(r[1])>=6
            if len(l2)==1 and not r[1]:
                st['sa']+=1; st['sa_st']+=len(g2)
                if len(g2)>=2: st['sa2']+=1
    return {k:round(v/ng,2) for k,v in st.items()}
for ps in (13,14,16,17): print(ps,sa(ps))
```