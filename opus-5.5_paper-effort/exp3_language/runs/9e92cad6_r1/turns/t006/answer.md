```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
outs={}
for j in range(6):
    U,s,Vt=np.linalg.svd(OV(0,j))
    outs[j]=U[:,:3]
    print(f"L0H{j} OV sv {s[:6]}")
    for k in range(2):
        u=U[:,k]; idx=np.argsort(-np.abs(u))[:5]
        print(f"   out dir {k}: top coords {idx} vals {u[idx]} ; in dir top coords {np.argsort(-np.abs(Vt[k]))[:5]} vals {Vt[k][np.argsort(-np.abs(Vt[k]))[:5]]}")
# key top dirs of L1
w=W["ln1_w.1"]
Ks=np.concatenate([hm("W_k",1,j)*w for j in range(6)],0)
U,s,Vt=np.linalg.svd(Ks); print("stacked L1 key sv", s[:6])
for k in range(3):
    v=Vt[k]; idx=np.argsort(-np.abs(v))[:6]; print(f" L1 key dir {k} coords {idx} vals {v[idx]}")
    for j in range(6): print(f"    overlap with L0H{j} out dirs", np.abs(outs[j].T@v))
print("ln1_b.1 big coords", np.argsort(-np.abs(W['ln1_b.1']))[:8], W['ln1_b.1'][np.argsort(-np.abs(W['ln1_b.1']))[:8]])
print("ln1_w.1 at those", W['ln1_w.1'][np.argsort(-np.abs(W['ln1_b.1']))[:8]])
```