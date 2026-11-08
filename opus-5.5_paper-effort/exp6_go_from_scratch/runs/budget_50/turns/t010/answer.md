```python
code=r'''
import json,numpy as np
def load(k):
    G=json.load(open('/work/pass_%d/games.json'%k)); B=np.load('/work/pass_%d/boards.npy'%k)
    return G,B
def show(bd):
    return '\n'.join(''.join('.XO'[v] for v in bd[r*9:(r+1)*9]) for r in range(9))
def showmany(bds):
    rows=[show(b).split('\n') for b in bds]
    return '\n'.join('  '.join(r[i] for r in rows) for i in range(9))
'''
open('/work/ana.py','w').write(code)
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
gs=[i for i,g in enumerate(G) if g['opponent']=='strong']
i=gs[6]; g=G[i]; n=len(g['moves'])
for m in g['moves'][-8:]: print(m['ply'],m['by'],m['move'],m.get('top5'))
print(showmany([B[i,k] for k in range(n-6,n)]))
```