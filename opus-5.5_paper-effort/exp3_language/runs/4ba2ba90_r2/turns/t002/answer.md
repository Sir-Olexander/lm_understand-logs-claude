```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
s=np.linalg.svd(W['W_proj.3'],compute_uv=False); print('W_proj.3', s[:45])
s=np.linalg.svd(W['W_o.3'],compute_uv=False); print('W_o.3', s[:45])
for l in range(3):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False); print(f'W_proj.{l}', s[:40])
hd=64
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    q=W['W_q.0'][sl]*W['ln1_w.0']; k=W['W_k.0'][sl]*W['ln1_w.0']
    s=np.linalg.svd(q.T@k,compute_uv=False); print('L0H%d QK(ln)'%j, s[:40])
print('ln1_w.0 sorted', np.sort(W['ln1_w.0'])[::-1][:80])
print('count ln1_w.0>0.1', (W['ln1_w.0']>0.1).sum(), '>0.3', (W['ln1_w.0']>0.3).sum())
```