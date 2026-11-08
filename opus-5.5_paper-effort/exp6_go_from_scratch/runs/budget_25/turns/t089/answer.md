```python
import json, numpy as np
d=json.load(open('/work/pass_13/games.json'))
for res in (1,-1):
    tops=[];nt=0;tot=0
    for g in d:
        if g['opponent']!='strong' or g['result']!=res: continue
        for m in g['moves']:
            if m['by']=='net' and m['top5']:
                tot+=1; tops.append(m['top5'][0][1])
                if m['move']!=m['top5'][0][0]: nt+=1
    print(res,'moves',tot,'not top1 frac',nt/tot,'mean top1',np.mean(tops))
```