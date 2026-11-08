```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmat(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
for l in range(4):
    w=W[f"ln1_w.{l}"]
    for j in range(6):
        Q=headmat("W_q",l,j)*w; K=headmat("W_k",l,j)*w
        M=K@Q.T   # 64x64, same nonzero eigs as Q^T K
        ev=np.linalg.eigvals(M)
        ev=ev[np.argsort(-np.abs(ev))]
        top=ev[:16]
        ang=np.degrees(np.angle(top))
        # fraction of sum of |ev| for real-positive, real-negative, complex
        a=np.angle(ev); m=np.abs(ev)
        pos=m[np.abs(a)<np.radians(20)].sum(); neg=m[np.abs(a)>np.radians(160)].sum(); cpx=m.sum()-pos-neg
        sym=np.linalg.norm((Q.T@K+K.T@Q)/2); asym=np.linalg.norm((Q.T@K-K.T@Q)/2)
        print(f"L{l}H{j} trace {np.trace(M):.2f} |sym| {sym:.2f} |asym| {asym:.2f} pos/neg/cpx {pos:.1f}/{neg:.1f}/{cpx:.1f}")
        print("   |ev|", np.abs(top)); print("   ang", ang.round(0))
```