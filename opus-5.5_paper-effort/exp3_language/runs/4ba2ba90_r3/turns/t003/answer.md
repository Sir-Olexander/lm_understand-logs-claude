```python
import numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for l in range(4):
    for n in ['W_o','W_proj']:
        s=np.linalg.svd(W[f'{n}.{l}'],compute_uv=False)
        print(f"{n}.{l}:", s[:40]); print("      ", s[40:80:2], "median %.3f"%np.median(s))
# shared output subspace of layer 3
Uo,so,_=np.linalg.svd(W['W_o.3'],full_matrices=False)
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for k in [10,20,24,26,28,30,35,40]:
    c=np.linalg.svd(Uo[:,:k].T@Up[:,:k],compute_uv=False)
    print(k, "overlap W_o.3 vs W_proj.3 top-k left sv cos:", np.round(c,2))
```