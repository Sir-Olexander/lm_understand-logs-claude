```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmat(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
# top right singular vectors of W_k (eff) for layers 1,2,3
for l in [1,2,3]:
    w=W[f"ln1_w.{l}"]; b=W[f"ln1_b.{l}"]
    tops=[]
    for j in range(6):
        K=headmat("W_k",l,j)*w[None,:]
        U,s,Vt=np.linalg.svd(K)
        tops.append(Vt[:2])
        Q=headmat("W_q",l,j)*w[None,:]
        # constant query from bias and key from bias
        qb=headmat("W_q",l,j)@b; kb=headmat("W_k",l,j)@b
        print(f"L{l}H{j} K sv {s[:4]}  |Wq b|={np.linalg.norm(qb):.2f} |Wk b|={np.linalg.norm(kb):.2f}")
    T=np.concatenate(tops,0)
    print(" overlap of top-2 key dirs across heads (svd of stacked):", np.linalg.svd(T,compute_uv=False))
```