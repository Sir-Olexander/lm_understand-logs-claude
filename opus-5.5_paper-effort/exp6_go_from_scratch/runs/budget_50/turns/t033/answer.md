```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
idx=[i for i,g in enumerate(G) if g['end']=='repetition']
for i in (idx[2],idx[4]):
    g=G[i]; n=len(g['moves'])
    print([(m['by'][0],m['move'],m.get('top5',[])[:2]) for m in g['moves'][-6:]])
    print(showmany([B[i,k] for k in range(n-7,n)]))
```