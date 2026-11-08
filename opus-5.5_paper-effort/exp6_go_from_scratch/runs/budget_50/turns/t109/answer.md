```python
import sys; sys.path.insert(0,'/work'); from ana import *
from collections import Counter
G,B=load(24)
fm=Counter(); res={}
for g in G:
    if g['net_color']!='black': continue
    p=g['moves'][0]['move']; fm[divmod(p,9)]+=1
    res.setdefault(min(p//9,p%9,8-p//9,8-p%9),[]).append(g['result']>0)
print(fm.most_common(12)); print({k:(len(v),sum(v)) for k,v in res.items()})
print(G[0]['moves'][0]['top5'])
```