```python
import json
from collections import Counter
for k in (4,5,6,8,9):
    G=json.load(open(f'/work/pass_{k}/games.json'))
    c=Counter()
    for g in G:
        if g['opponent']!='strong' or g['net_color']!='black': continue
        t = g['moves'][0]['move']==40
        c[(t,g['result'])]+=1
    print(k,'tengen W/L',c[(True,1)],c[(True,-1)],' other W/L',c[(False,1)],c[(False,-1)])
```