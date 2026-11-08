```python
import sys; sys.path.insert(0,'/work'); from ana import *
import numpy as np
X=np.load('/work/pass_13/inputs.npy'); lg=np.load('/work/pass_13/logits.npy')
x=X[2].reshape(81,3); bd=(x[:,0]-x[:,1]).astype(int)
print(show(bd)); print('logit(1,1)',lg[2,10],'pass',lg[2,81])
for c,st,l in groups(bd):
    if 10 in [q for s_ in st for q in NBL[s_]] : print(c,len(st),sorted(st)[:12],sorted(l))
```