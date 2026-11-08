```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
U,s,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
print("sv", s[:40])
def enr(M): 
    M=M/np.linalg.norm(M); return (np.linalg.norm(U[:,:40].T@M,axis=1)**2)*384
for j in range(6):
    print(f"L3H{j}", enr(OV(3,j)))
print("MLP2  ", enr(W['W_proj.2']))
print("MLP1  ", enr(W['W_proj.1']))
for j in [3]:
    print(f"L2H{j}", enr(OV(2,j)))
# participation ratio of neuron loadings for each direction
pr=[(np.sum(Vt[k]**2)**2)/np.sum(Vt[k]**4)/1536 for k in range(40)]
print("neuron PR frac", np.array(pr))
```