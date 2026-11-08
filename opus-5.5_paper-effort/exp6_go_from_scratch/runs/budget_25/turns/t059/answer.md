```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, collections
d=json.load(open('/work/pass_8/games.json'))
B=np.load('/work/pass_8/boards.npy')
tob=lambda b:[1 if x==1 else (2 if x==-1 else 0) for x in b]
def showme(b,me):
    ch={0:'.',me:'X',3-me:'O'}
    return '\n'.join(''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9))
# For passes by net: evaluate if any own group in atari or opp capture existed (true), and count 'dame' regions
st=collections.Counter(); ex=[]
for gi,g in enumerate(d):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']!=81 or k==0: continue
        prev=tob(B[gi,k-1])
        gid,groups=G.all_groups(prev)
        capav=any(len(l)==1 and c!=me for gg,l,c in groups)
        savav=any(len(l)==1 and c==me for gg,l,c in groups)
        st['passes']+=1; st['capav']+=capav; st['savav']+=savav
        # dame points: empty points adjacent to both colors
        dame=[s for s in range(81) if prev[s]==0 and any(prev[n]==me for n in G.NB[s]) and any(prev[n]==3-me for n in G.NB[s])]
        if dame: st['with_dame']+=1; st['dame_pts']+=len(dame)
        if (dame or capav or savav) and len(ex)<5 and g['result']==-1: ex.append((gi,k,g['margin'],capav,savav,[divmod(x,9) for x in dame]))
print(st)
for gi,k,mg,ca,sa,dm in ex:
    me=1 if d[gi]['net_color']=='black' else 2
    print(gi,k,mg,'cap',ca,'save',sa,'dame',dm); print(showme(tob(B[gi,k-1]),me))
```