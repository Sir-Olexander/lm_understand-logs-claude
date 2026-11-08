```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
for gi,g in enumerate(G):
    if g['result']==-1 or g['end']!='two passes':
        n=len(g['moves'])
        print(gi,g['opponent'],g['net_color'],g['result'],g['margin'],g['end'],n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],
              'netpasses',sum(1 for m in g['moves'] if m['by']=='net' and m['move']==81))
```