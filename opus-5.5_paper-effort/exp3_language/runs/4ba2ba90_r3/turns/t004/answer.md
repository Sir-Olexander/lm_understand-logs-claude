```python
import numpy as np
from scipy.stats import kurtosis
np.set_printoptions(linewidth=200, precision=2, suppress=True)
P=W['W_proj.3']
U,s,Vt=np.linalg.svd(P,full_matrices=False)
k=36
print("sv:",s[:k])
print("kurtosis of neuron loadings (Vt rows):",np.round(kurtosis(Vt[:k],axis=1),1))
# fraction of loading energy in top 1% neurons
E=Vt[:k]**2
top=np.sort(E,axis=1)[:,::-1]
print("energy share of top 15 neurons:",np.round(top[:,:15].sum(1),2))
# also W_fc.3 rows of these neurons: does input-side align?
# ln2_w.3 for reference; and same for W_o.3 right side? left side alignment of W_o.3 with U
Uo,so,_=np.linalg.svd(W['W_o.3'],full_matrices=False)
print("energy of W_o.3 in U-dirs (per dir, norm of U_i^T W_o.3):",np.round(np.linalg.norm(U[:,:k].T@W['W_o.3'],axis=1),2))
R=np.linalg.qr(np.random.default_rng(0).normal(size=(384,10)))[0]
print("random dirs:",np.round(np.linalg.norm(R.T@W['W_o.3'],axis=1),2))
for l in range(3):
    print(f"W_proj.{l} norm on U dirs:",np.round(np.linalg.norm(U[:,:30].T@W[f'W_proj.{l}'],axis=1),1), " random:",np.round(np.linalg.norm(R.T@W[f'W_proj.{l}'],axis=1),1)[:5])
```