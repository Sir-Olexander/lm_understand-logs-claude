```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A=q.T@k
U0,s0,Vt0=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v)
OVp=o@v
Uout,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=Uout[:,:40]
for kT in [20,28,40]:
    T=Vt2[:kT].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
    q,k,v,o=head(3,1); OV=o@v
    print(f"T{kT}: L3H1 per-dir copy energy:",np.linalg.norm(U.T@OV@T,axis=1))
rng=np.random.default_rng(0); R=np.linalg.qr(rng.normal(size=(384,28)))[0]
q,k,v,o=head(3,1); OV=o@v
print("random-28d input:          ",np.linalg.norm(U.T@OV@R,axis=1))
# prev-token-out space read by L3H1 V? and the induction-prediction: tokens matched by QK
T=Vt2[:28].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
M=T.T@(q.T@k)@OVp@T
print("induction match matrix sym eig:",np.linalg.eigvalsh((M+M.T)/2)[::-1])
```