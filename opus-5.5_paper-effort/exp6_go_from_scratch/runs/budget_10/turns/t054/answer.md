```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_6/games.json'))
B=np.load('/work/pass_6/boards.npy')
for gi,g in enumerate(G):
    if g['result']==-1 or g['end']!='two passes':
        n=len(g['moves'])
        print(gi,g['opponent'],g['net_color'],g['result'],g['margin'],g['end'],n,'lastby',g['moves'][-1]['by'],'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
tot=0; bl=[]
for gi,g in enumerate(G):
    netcol = -1 if g['net_color']=='white' else 1
    mv=g['moves']
    for k in range(len(mv)-1):
        m=mv[k]
        if m['by']!='net' or m['move']==81: continue
        tot+=1; p=m['move']
        if B[gi,k+1][p]!=netcol:
            lost=int(((B[gi,k]==netcol)&(B[gi,k+1]!=netcol)).sum())
            bl.append((gi,k,lost,m['top5'][0][1]))
print(tot,len(bl),sorted(x[2] for x in bl))
print(bl)
from collections import Counter
print(Counter(divmod(g['moves'][0]['move'],9) for g in G if g['net_color']=='black').most_common(5))
```