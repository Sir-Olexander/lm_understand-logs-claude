```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
gi=49; g=G[gi]
# find plies where net lost many stones
prev=None
for k in range(len(g['moves'])):
    bd=B[gi,k]
    netcol = -1 if g['net_color']=='white' else 1
    cnt=(bd==netcol).sum()
    if prev is not None and cnt<prev-2: print('ply',k,'net stones',prev,'->',cnt)
    prev=cnt
for k in [30,40,50,56,60]:
    print(k); print(show(B[gi,k]))
for m in g['moves'][40:62]:
    mv=m['move']; print(m['ply'],m['by'][:3],divmod(mv,9) if mv<81 else 'P',[(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])][:3])
```