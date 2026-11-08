```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_8/games.json'))
B=np.load('/work/pass_8/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
st=collections.Counter()
ex=[]
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']!=81 or k==0: continue
        prev=tob(B[gi,k-1])
        # empty regions bordering both colors (dame-ish)
        seen=set(); dame=0; dameregions=0
        for s in range(81):
            if prev[s]==0 and s not in seen:
                st_=[s]; reg={s}; border=set()
                while st_:
                    x=st_.pop()
                    for n in G.NB[x]:
                        if prev[n]==0 and n not in reg: reg.add(n); st_.append(n)
                        elif prev[n]: border.add(prev[n])
                seen|=reg
                if len(border)==2: dame+=len(reg); dameregions+=1
        st['passes']+=1
        nst=sum(1 for v in prev if v)
        if dameregions: st['pass_with_neutral_region']+=1; st['neutral_pts']+=dame
        prevm=g['moves'][k-1]['move']
        if prevm==81: st['after_opp_pass']+=1
        if dameregions and len(ex)<4: ex.append((gi,k,g['margin'],nst,[(divmod(a,9),p) for a,p in m['top5'][:3]]))
print(st)
for gi,k,mg,nst,top in ex:
    print(gi,k,mg,nst,top); print(G.show(tob(B[gi,k-1])))
```