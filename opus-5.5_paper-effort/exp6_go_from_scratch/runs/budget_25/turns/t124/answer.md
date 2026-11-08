```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def line(s):
    r,c=divmod(s,9); return min(r,c,8-r,8-c)+1
cat=collections.Counter()
for ps in (17,19,20):
    d=json.load(open('/work/pass_%d/games.json'%ps)); B=np.load('/work/pass_%d/boards.npy'%ps)
    for gi,g in enumerate(d):
        me=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            p=m['move']
            if line(p)!=1: continue
            prev=tob(B[gi,k-1]); r=G.play(prev,p,me)
            if not r or not r[1]: continue
            ne=sum(1 for n in G.NB[p] if prev[n]==0)
            own=[n for n in G.NB[p] if prev[n]==me]
            W=0
            for q in own:
                nl=len(G.group(prev,q)[1]); W+=min(nl,3)-2
            uc=max(0,4-3*ne-2.5*W-1.5*len(own))
            kadj=sum(1 for n in G.NB[p] if prev[n]==3-me and len(G.group(prev,n)[1])==1)
            cat[(round(uc,1),kadj,'big' if len(r[1])>=5 else 'small')]+=1
for k,v in sorted(cat.items(),key=lambda x:-x[1])[:14]: print('Uc,kadj,size',k,v)
print(sum(cat.values()))
```