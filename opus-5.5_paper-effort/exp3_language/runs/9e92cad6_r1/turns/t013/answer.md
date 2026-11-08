```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
U,s,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
ks=[16,18,20,21,22,23,24,25,26,27,28,29,30,32,35,40,50]
print("k      ", ks)
for j in range(6):
    O=OV(3,j); tot=np.linalg.norm(O)**2
    print(f"L3H{j}", [round(np.linalg.norm(U[:,:k].T@O)**2/tot,3) for k in ks])
# also the W_o.3 (all heads) column space
O=W['W_o.3']; tot=np.linalg.norm(O)**2
print("W_o.3 ", [round(np.linalg.norm(U[:,:k].T@O)**2/tot,3) for k in ks])
# baseline random
print("random", [round(k/384,3) for k in ks])
# combined writer svd
C=np.concatenate([W['W_proj.3']/np.linalg.norm(W['W_proj.3'])]+[OV(3,j)/np.linalg.norm(OV(3,j)) for j in range(6)],1)
print("combined writers sv", np.linalg.svd(C,compute_uv=False)[:45])
```