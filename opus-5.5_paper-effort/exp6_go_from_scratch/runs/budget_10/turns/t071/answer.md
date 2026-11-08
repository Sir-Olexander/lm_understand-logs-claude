```python
import json, numpy as np
for k in (4,5,6,8,9):
    G=json.load(open(f'/work/pass_{k}/games.json'))
    S=[g for g in G if g['opponent']=='strong']
    close=[g for g in S if abs(g['margin'])<8]
    cw=sum(1 for g in close if g['result']==1)
    rep=sum(1 for g in S if g['end']=='repetition' and g['result']==-1)
    blow=sum(1 for g in S if g['margin']>40)
    lostbig=sum(1 for g in S if g['margin']<-8)
    length=np.mean([len(g['moves']) for g in S])
    print(k,'close',len(close),'closewin',cw,'replosses',rep,'wins>40',blow,'losses<-8',lostbig,'len',round(length,1))
```