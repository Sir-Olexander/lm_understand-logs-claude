```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
cases=[]
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net': continue
        prev=tob(B[gi,k-1]) if k>0 else [0]*81
        gid,groups=G.all_groups(prev)
        best=None
        for gg,l,c in groups:
            if len(l)==1 and c==me and len(gg)>=2:
                lib=next(iter(l)); r=G.play(prev,lib,me)
                if r:
                    g2,l2=G.group(r[0],lib)
                    if len(l2)>=2: best=(len(gg),lib,len(l2))
        if best:
            cases.append((gi,k,best,m['move'],m['top5'][:3]))
print(len(cases))
cnt=collections.Counter()
for gi,k,best,mv,top in cases:
    prev=tob(B[gi,k-1]); me=1 if d[gi]['net_color']=='black' else 2
    if mv==81: cnt['pass']+=1; continue
    nb,cap,_=G.play(prev,mv,me)
    if cap: cnt['captured_instead']+=1
    elif mv==best[1]: cnt['saved']+=1
    else: cnt['other']+=1
print(cnt)
ch={0:'.',1:'X',2:'O'}
shown=0
for gi,k,best,mv,top in cases:
    prev=tob(B[gi,k-1]); me=1 if d[gi]['net_color']=='black' else 2
    if mv==81 or G.play(prev,mv,me)[1] or mv==best[1]: continue
    print(gi,k,'me',ch[me],'group size',best[0],'lib',divmod(best[1],9),'libs after',best[2],'played',divmod(mv,9),[(divmod(a,9),p) for a,p in top])
    print(G.show(prev)); shown+=1
    if shown>=6: break
```