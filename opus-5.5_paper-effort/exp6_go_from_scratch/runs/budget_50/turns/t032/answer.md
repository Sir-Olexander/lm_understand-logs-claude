```python
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(7)
for i,g in enumerate(G):
    if g['end']=='repetition':
        mv_=g['moves']; last=mv_[-1]
        print(g['opponent'],g['net_color'],g['result'],g['margin'],len(mv_),'last by',last['by'],last['move'],[m['move'] for m in mv_[-8:]])
```