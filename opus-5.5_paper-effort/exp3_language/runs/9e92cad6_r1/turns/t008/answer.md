```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w=W["ln1_w.0"]
for j in [3,0,5]:
    s=np.linalg.svd(hm("W_v",0,j)*w,compute_uv=False); print(f"V eff L0H{j}", s[:40])
    s=np.linalg.svd(hm("W_v",0,j),compute_uv=False); print(f"V raw L0H{j}", s[:40])
    s=np.linalg.svd(hm("W_o",0,j)@(hm("W_v",0,j)*w),compute_uv=False); print(f"OV L0H{j}", s[:40])
    s=np.linalg.svd(hm("W_q",0,j)*w,compute_uv=False); print(f"Q eff L0H{j}", s[:40])
# stacked V of all L0 heads
Vs=np.concatenate([hm("W_v",0,j)*w for j in [0,3,5]],0)
print("stacked V eff 0,3,5", np.linalg.svd(Vs,compute_uv=False)[:45])
```