```python
import numpy as np
np.set_printoptions(linewidth=200, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for h in [2,3]:
    q,k,v,o=head(0,h); A=q.T@k
    ev=np.linalg.eigvals(A); ev=ev[np.argsort(-np.abs(ev))][:40]
    print(f"L0H{h} top eigs |.|:",np.abs(ev)); print("   arg/pi:",np.angle(ev)/np.pi)
    OV=o@v; ev=np.linalg.eigvals(OV); ev=ev[np.argsort(-np.abs(ev))][:20]
    print("   OV eig |.|",np.abs(ev)); print("   OV arg/pi",np.angle(ev)/np.pi)
# Q side vs K side subspace overlap for L0H2
q,k,v,o=head(0,2); U,s,Vt=np.linalg.svd(q.T@k)
c=np.linalg.svd(U[:,:20].T@Vt[:20].T,compute_uv=False); print("L0H2 Qside vs Kside (20d) cos:",c)
# overlap of L0H2 QK space with its own OV input space
U2,s2,Vt2=np.linalg.svd(o@v); 
print("L0H2 K-side vs OV-input (20d):",np.linalg.svd(Vt[:20]@Vt2[:20].T,compute_uv=False))
print("L0H2 Q-side vs OV-input (20d):",np.linalg.svd(U[:,:20].T@Vt2[:20].T,compute_uv=False))
```