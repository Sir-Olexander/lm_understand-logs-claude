```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,_=pickle.load(open('/work/v13_units.pkl','rb'))
d=json.load(open('/work/pass_17/games.json')); B=np.load('/work/pass_17/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
for gi in (70,75,123,47,12):
    g=d[gi]; me=1 if g['net_color']=='black' else 2
    k=max(i for i,m in enumerate(g['moves']) if m['by']=='net')
    prev=tob(B[gi,k-1])
    print(gi,g['net_color'],g['margin'])
    print(showme(prev,me))
    for s in range(81):
        if prev[s]!=0: continue
        if any(prev[n]==me for n in G.NB[s]) and any(prev[n]==3-me for n in G.NB[s]):
            r=G.play(prev,s,me)
            if r is None: print('  dame',divmod(s,9),'illegal'); continue
            g2,l2=G.group(r[0],s)
            ne=sum(1 for n in G.NB[s] if prev[n]==0)
            print('  dame',divmod(s,9),'libs after',len(l2),'cap',len(r[1]),'ne',ne)
```