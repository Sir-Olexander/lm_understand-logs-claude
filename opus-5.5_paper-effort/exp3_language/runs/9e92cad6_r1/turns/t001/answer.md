```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    print("layer", l)
    for nm in ["ln1_w","ln1_b","ln2_w","ln2_b"]:
        v = W[f"{nm}.{l}"]; print(f"  {nm}: mean {v.mean():.3f} std {v.std():.3f} min {v.min():.3f} max {v.max():.3f}")
    for nm in ["W_q","W_k","W_v","W_o"]:
        M = W[f"{nm}.{l}"]
        for j in range(6):
            Mh = M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
            s = np.linalg.svd(Mh, compute_uv=False)
            print(f"  {nm} h{j}: fro {np.linalg.norm(Mh):.2f} top sv {s[:6]} sv10 {s[9]:.3f} sv32 {s[31]:.3f}")
    for nm in ["W_fc","W_proj"]:
        s = np.linalg.svd(W[f"{nm}.{l}"], compute_uv=False)
        print(f"  {nm}: fro {np.linalg.norm(W[f'{nm}.{l}']):.2f} top sv {s[:8]} sv50 {s[49]:.3f} sv200 {s[199]:.3f}")
```