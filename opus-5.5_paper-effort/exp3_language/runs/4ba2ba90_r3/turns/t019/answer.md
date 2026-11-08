```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,k): U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
OVb=sum(head(0,h)[3]@head(0,h)[2] for h in [0,1,3,4,5]); Bs=orth(OVb,3)
q,k,v,o=head(0,2); OVp=o@v; P0=orth(OVp,24)
Uout,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
idx=[0,1,2,22,23,24,25,26,27,28,29,30,31,33,34,35]
for i in idx:
    u=Uout[:,i]
    print(f"U{i} (sv {sp[i]:.1f}): |cos| with b-out {np.linalg.norm(Bs.T@u):.2f}, prevOut {np.linalg.norm(P0.T@u):.2f}, W_proj.0 top10 {np.linalg.norm(orth(W['W_proj.0'],10).T@u):.2f}, ones {abs(u.sum())/np.sqrt(384):.2f}")
# largest-norm directions of ln biases (massive dims?)
for n in ['ln1_b.1','ln1_b.2','ln1_b.3','ln2_b.3']:
    b=W[n]; print(n,"top coords",np.argsort(-np.abs(b))[:6],np.round(b[np.argsort(-np.abs(b))[:6]],2))
print("U0 top coords",np.argsort(-np.abs(Uout[:,0]))[:6],np.round(Uout[np.argsort(-np.abs(Uout[:,0]))[:6],0],2))
print("Bs0 top coords",np.argsort(-np.abs(Bs[:,0]))[:6],np.round(Bs[np.argsort(-np.abs(Bs[:,0]))[:6],0],2))
```