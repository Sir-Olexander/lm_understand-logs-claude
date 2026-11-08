```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A=q.T@k
U0,s0,Vt0=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v)
print("L0H2 OV full sv:",s2[:60])
T=Vt2[:40].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
Uout,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); Uout=Uout[:,:29]
for h in range(6):
    q,k,v,o=head(3,h); OV=o@v
    s=np.linalg.svd(Uout.T@OV@T,compute_uv=False)
    s_all=np.linalg.svd(Uout.T@OV,compute_uv=False)
    print(f"L3H{h} U^T OV T sv:",s[:32]); print("        U^T OV (all inputs) sv:",s_all[:29])
# V-input reading of T by L3 heads
for h in range(6):
    q,k,v,o=head(3,h)
    print(f"L3H{h} V reads T(40d) %.2f  Ppos %.2f rand %.2f"%(np.linalg.norm(v@T)/np.sqrt(40),np.linalg.norm(v@Ppos)/np.sqrt(24),np.linalg.norm(v)/np.sqrt(384)))
```