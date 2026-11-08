```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
for gi in [34,18]:
    g=G[gi]; netcol = -1 if g['net_color']=='white' else 1
    prev=None; events=[]
    for k in range(len(g['moves'])):
        cnt=(B[gi,k]==netcol).sum()
        if prev is not None and cnt<prev-1: events.append((k,prev-cnt+1))
        prev=cnt
    print('game',gi,g['net_color'],g['margin'],events)
    k,sz=max(events,key=lambda t:t[1])
    for kk in [k-12,k-6,k-1]:
        print(kk); print(show(B[gi,kk]))
    for m in g['moves'][k-12:k+1]:
        mv=m['move']; print(m['ply'],m['by'][:3],divmod(mv,9) if mv<81 else 'P',[(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])][:3])
```