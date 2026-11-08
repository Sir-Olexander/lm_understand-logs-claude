```python
import json, numpy as np, collections
d=json.load(open('/work/pass_20/games.json'))
fm=collections.Counter(); res=collections.defaultdict(list)
for g in d:
    if g['net_color']!='black': continue
    m=g['moves'][0]; fm[divmod(m['move'],9)]+=1; res[divmod(m['move'],9)].append(g['result'])
for k,v in fm.most_common(): print(k,v,np.mean(res[k]))
print(d[0]['moves'][0]['top5'])
# white: our response to opp first move
wr=collections.Counter()
for g in d:
    if g['net_color']!='white': continue
    wr[(divmod(g['moves'][0]['move'],9),divmod(g['moves'][1]['move'],9))]+=1
print(wr.most_common(8))
```