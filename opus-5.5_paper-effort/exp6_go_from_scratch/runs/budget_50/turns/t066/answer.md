```python
import numpy as np, sys; sys.path.insert(0,'/work'); import build2
from netlib import *
d='/work/pass_14/'
A={k:np.load(d+'act_L%d.npy'%k)[0] for k in range(1,7)}
n=build2.build(build2.default_prm(),inp,mv)
print(n.blk)
p=10
for k in range(5,6):
    print({nm: A[6][n.blk[5][nm]+p] for nm in ['Eye','Pos','U2','U3','Terr']})
for q in [1,9,11,19,0,2]:
    print(q,{nm:A[5][n.blk[4][nm]+q] for nm in ['S','O','NS','NO','US','UO']}, 'Ev',A[2][n.blk[1]['EvS']+q],'E3',A[2][n.blk[1]['E3S']+q],'LS',A[1][n.blk[0]['LS']+q])
```