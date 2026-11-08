```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
print("ln1_b at outliers per layer:", [W[f"ln1_b.{l}"][out] for l in range(4)])
print("ln1_w at outliers per layer:", [W[f"ln1_w.{l}"][out] for l in range(4)])
for j in [1,2,4,5]:
    O=OV(0,j); print(f"L0H{j} OV rows at outliers (norms):", np.linalg.norm(O[out],axis=1), " total", np.linalg.norm(O))
for l in [1,2,3]:
    w=W[f"ln1_w.{l}"]; b=W[f"ln1_b.{l}"]
    for j in range(6):
        Q=hm("W_q",l,j)*w; K=hm("W_k",l,j)*w
        qb=hm("W_q",l,j)@b
        keybias=(qb@K)[out]     # key preference from constant query in outlier coords
        B=(Q.T@K)[np.ix_(out,out)]
        print(f"L{l}H{j} keybias(out) {keybias}  B_out diag {np.diag(B)}  B[21,99] {B[0,3]:.2f} B[99,21] {B[3,0]:.2f}")
```