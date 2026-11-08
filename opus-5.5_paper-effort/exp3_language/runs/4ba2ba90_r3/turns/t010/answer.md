```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
Uout,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=Uout[:,:36]
print("W_proj.3 sv     :",sp[:36])
for h in range(6):
    q,k,v,o=head(3,h)
    print(f"L3H{h} out energy:",np.linalg.norm(U.T@(o@v),axis=1))
print("W_o.3 energy    :",np.linalg.norm(U.T@W['W_o.3'],axis=1))
for l in range(3):
    print(f"W_proj.{l} energy :",np.linalg.norm(U.T@W[f'W_proj.{l}'],axis=1))
    print(f"W_o.{l} energy    :",np.linalg.norm(U.T@W[f'W_o.{l}'],axis=1))
# how W_fc.3 reads these output dirs (input side same space?)
print("W_fc.3 read     :",np.linalg.norm(W['W_fc.3']@U,axis=0))
print("ln1_b/ln2_b.3 proj:",U.T@W['ln2_b.3'])
```