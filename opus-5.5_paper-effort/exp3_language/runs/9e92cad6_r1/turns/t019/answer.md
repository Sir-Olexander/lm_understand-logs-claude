```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
w0=W["ln1_w.0"]
U,s,Vt=np.linalg.svd(np.concatenate([hm("W_v",0,3)*w0, hm("W_v",0,0)*w0],0),full_matrices=False)
T=Vt[:22].T
srcs=[("L0H3",OV(0,3)),("L0H0",OV(0,0))]+[(f"L1H{j}",OV(1,j)) for j in range(6)]+[(f"L2H{j}",OV(2,j)) for j in range(6)]
print("matching score = trace/sum|eig| of T^T Q^T K OV_src T  (positive ~ induction-like)")
print("tgt   "+" ".join(f"{n:>6}" for n,_ in srcs))
for l in [2,3]:
    w=W[f"ln1_w.{l}"]
    for j in range(6):
        Q=hm("W_q",l,j)*w; K=hm("W_k",l,j)*w
        row=[]
        for n,O in srcs:
            if int(n[1])>=l: row.append("     -"); continue
            A=T.T@Q.T@K@O@T; ev=np.linalg.eigvals(A)
            row.append(f"{np.trace(A):6.2f}")
        print(f"L{l}H{j} "+" ".join(row))
# also V-side: does L1 head OV copy tokens? measure ||OV T|| relative
print("token-copy strength ||OV_src T||_F:", [(n, round(float(np.linalg.norm(O@T)),2)) for n,O in srcs])
```