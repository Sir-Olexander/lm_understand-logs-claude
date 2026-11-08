```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A0=q.T@k; OVp=o@v
U0,s0,Vt0=np.linalg.svd(A0); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(OVp); T=Vt2[:48].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
S=np.zeros((T.shape[1],)*2)
for h in range(6):
    q,k,v,o=head(1,h); M=T.T@(q.T@k)@T; S+= (M+M.T)/2
    ev=np.linalg.eigvalsh((M+M.T)/2); print(f"L1H{h} sym(tt) eig:",np.round(ev[:34],2))
ev=np.linalg.eigvalsh(S); print("sum L1 sym(tt) eig:",np.round(ev,2))
# same with full space (no T restriction): symmetric part of summed L1 QK
Sf=sum(((head(1,h)[0].T@head(1,h)[1])+(head(1,h)[0].T@head(1,h)[1]).T)/2 for h in range(6))
evf=np.linalg.eigvalsh(Sf); print("full-space sum L1 sym QK most negative:",np.round(evf[:45],2)); print("most positive:",np.round(evf[::-1][:20],2))
```