```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
w0=W["ln1_w.0"]
V=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
def frac(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
U,s,_=np.linalg.svd(OV(0,3))
print("O3 out dirs: sv and V0V3-enrichment")
print(s[:30]); print(np.array([frac(V,U[:,k]) for k in range(30)]))
U0,s0,_=np.linalg.svd(OV(0,0))
print("O0:", s0[:20]); print(np.array([frac(V,U0[:,k]) for k in range(20)]))
# also other readers that read only x0: all L0 Q/K
QK=np.concatenate([hm(n,0,j)*w0 for n in ["W_q","W_k"] for j in range(6)],0)
print("O3 QK-enr:", np.array([frac(QK,U[:,k]) for k in range(30)]))
```