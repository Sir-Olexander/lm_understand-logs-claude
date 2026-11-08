```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d=384; hd=64
for l in range(4):
    print(f"--- layer {l}")
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        a=W[f'{nm}.{l}']; print(nm, 'mean %.3f std %.3f min %.3f max %.3f'%(a.mean(),a.std(),a.min(),a.max()))
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        a=W[f'{nm}.{l}']; s=np.linalg.svd(a,compute_uv=False)
        print(nm, 'fro %.2f'%np.linalg.norm(a), 'top sv', s[:8], 'sv[30] %.3f sv[100] %.3f median %.3f'%(s[30],s[100],np.median(s)))
    # per-head norms
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        q=W[f'W_q.{l}'][sl]; k=W[f'W_k.{l}'][sl]; v=W[f'W_v.{l}'][sl]; o=W[f'W_o.{l}'][:,sl]
        qk=(q.T@k); ov=o@v
        sqk=np.linalg.svd(qk,compute_uv=False); sov=np.linalg.svd(ov,compute_uv=False)
        print(f'  H{j} |q| %.2f |k| %.2f |v| %.2f |o| %.2f  QK sv %s  OV sv %s'%(np.linalg.norm(q),np.linalg.norm(k),np.linalg.norm(v),np.linalg.norm(o),sqk[:4],sov[:4]))
```