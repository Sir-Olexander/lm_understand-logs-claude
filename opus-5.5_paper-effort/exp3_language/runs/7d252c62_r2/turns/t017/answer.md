```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=2, suppress=True)
U,S,Vt=svd(W['W_v.0'])
w0=W['ln1_w.0']; w2=W['ln2_w.0']
mats={'fc0*w2':W['W_fc.0']*w2,'fc0raw':W['W_fc.0'],'q0raw':W['W_q.0'],'k0raw':W['W_k.0'],'q1raw':W['W_q.1'],'k1raw':W['W_k.1'],'v1raw':W['W_v.1'],'fc1raw':W['W_fc.1']}
rng=np.random.default_rng(0)
for nm,M in mats.items():
    e=np.array([norm(M@Vt[i])**2 for i in range(384)])
    e=e/np.median(e)
    print(f"{nm:7}", np.round(e[:45],1))
# also: energy of W_v.0 right sing vectors in ln1_w.0 terms
print("w0-weighted norms", np.round([norm(Vt[i]*w0) for i in range(45)],2))
```