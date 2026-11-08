```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j):
    return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
srcs={}
for l in range(3):
    for j in range(6): srcs[f"L{l}H{j}"]=OV(l,j)
    srcs[f"MLP{l}"]=W[f"W_proj.{l}"]
for kind in ["W_q","W_k","W_v"]:
    print("=== ",kind,"composition (x100), baseline ~5.1")
    names=list(srcs.keys())
    print("tgt\\src "+" ".join(f"{n:>7}" for n in names))
    for l in range(1,4):
        for j in range(6):
            T=hm(kind,l,j)*W[f"ln1_w.{l}"]
            row=[]
            for n in names:
                sl=int(n[1]) if n.startswith("L") else int(n[3])
                if sl>=l: row.append("      -"); continue
                row.append(f"{100*comp(T,srcs[n]):7.1f}")
            print(f"L{l}H{j}   "+" ".join(row))
```