```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(8)
idx=[i for i,g in enumerate(G) if g['end']=='repetition']
for i in idx:
    g=G[i]; print(g['net_color'],g['result'],g['margin'],g['moves'][-1]['by'],[m['move'] for m in g['moves'][-6:]])
i=[i for i in idx if G[i]['moves'][-1]['by']=='net'][0]; g=G[i]; n=len(g['moves'])
print([(m['by'][0],m['move'],(m.get('top5') or [[0,0]])[:2]) for m in g['moves'][-6:]])
print(showmany([B[i,k] for k in range(n-7,n)]))
```