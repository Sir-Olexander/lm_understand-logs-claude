```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for nm in ["W_fc.0","W_fc.1","W_proj.0","W_proj.3","W_proj.2"]:
    s = np.linalg.svd(W[nm], compute_uv=False)
    print(nm, s[:120])
# layer 0 combined reader
l=0
R = np.concatenate([W['W_q.0'],W['W_k.0'],W['W_v.0']],0)*W['ln1_w.0'][None,:]
s = np.linalg.svd(R, compute_uv=False); print("attn reader L0 (with ln_w)", s[:100])
R2 = W['W_fc.0']*W['ln2_w.0'][None,:]
s = np.linalg.svd(R2, compute_uv=False); print("fc reader L0 (with ln_w)", s[:100])
print("sorted ln1_w.0", np.sort(W['ln1_w.0'])[::-1][:80])
print("sorted ln1_w.0 low", np.sort(W['ln1_w.0'])[:40])
```