```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
w=W["ln1_w.3"]
Q=hm("W_q",3,3)*w; K=hm("W_k",3,3)*w
A=Q.T@K@OV(0,3)   # 384x384: maps token-ish x (at key pos j-1) -> query space
ev=np.linalg.eigvals(A); ev=ev[np.argsort(-np.abs(ev))]
print("eig |.|", np.abs(ev[:40])); print("angle", np.degrees(np.angle(ev[:40])).round(0))
# random comparison: a non-induction head
for (l,j) in [(3,0),(2,4),(1,1)]:
    Qx=hm("W_q",l,j)*W[f"ln1_w.{l}"]; Kx=hm("W_k",l,j)*W[f"ln1_w.{l}"]
    e=np.linalg.eigvals(Qx.T@Kx@OV(0,3)); e=e[np.argsort(-np.abs(e))]
    print(f"L{l}H{j} eig |.|", np.abs(e[:20]).round(2), "ang", np.degrees(np.angle(e[:20])).round(0))
# singular values of the L0H3 OV restricted after removing outlier coords
s=np.linalg.svd(OV(0,3),compute_uv=False); print("OV03 sv", s[:40])
# L3H3 OV
s=np.linalg.svd(OV(3,3),compute_uv=False); print("OV33 sv", s[:40])
s=np.linalg.svd(OV(3,4),compute_uv=False); print("OV34 sv", s[:40])
```