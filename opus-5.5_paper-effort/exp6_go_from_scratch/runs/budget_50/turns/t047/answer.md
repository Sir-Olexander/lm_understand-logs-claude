```python
import numpy as np, sys; sys.path.insert(0,'/work'); from ana import *
d='/work/pass_10/'
X=np.load(d+'inputs.npy'); lg=np.load(d+'logits.npy'); L=np.load(d+'legal.npy')
A={k:np.load(d+'act_L%d.npy'%k) for k in range(1,7)}
x=X[0].reshape(81,3); bd=(x[:,0]-x[:,1]).astype(int)
print(show(bd)); print('legal71',L[0,71],'logit71',lg[0,71],'max',lg[0].max())
for p in [35,44,53,62]:
    print(p,[ (A[k][0,162+p],A[k][0,324+p]) for k in (3,4,5)], 'Ev',A[2][0,162+p],A[2][0,324+p])
print('L6 at 71: Eye,Pos,U2,U3,Terr',[A[6][0,k*81+71] for k in range(5)])
```