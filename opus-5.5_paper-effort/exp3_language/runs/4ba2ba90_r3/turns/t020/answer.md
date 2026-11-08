```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def posr(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/(np.abs(ev).sum()+1e-12)
q,k,v,o=head(0,2); A0=q.T@k; OVp=o@v
U0,s0,Vt0=np.linalg.svd(A0); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(OVp); T=Vt2[:28].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
# normalise OVp so that its T->out map has comparable scale: use orthonormal-ized version
Qp,Rp=np.linalg.qr(OVp@T)   # prev-token rep basis (384x28) ; Qp = OVp T Rp^-1
for l in [1,2,3]:
    row=[]
    for h in range(6):
        q,k,v,o=head(l,h); A=q.T@k
        Mprev=(OVp@T).T@A@T          # query prevtok == key tok  -> attend to i-1 (content route)
        Mind=T.T@A@(OVp@T)           # query tok == key prevtok  -> induction
        Mtt=T.T@A@T                  # same token
        Mpp=(OVp@T).T@A@(OVp@T)      # same previous token
        row.append(f"H{h}: tt {posr(Mtt):+.2f}/{np.linalg.norm(Mtt):.1f} prev {posr(Mprev):+.2f}/{np.linalg.norm(Mprev):.2f} ind {posr(Mind):+.2f}/{np.linalg.norm(Mind):.2f} pp {posr(Mpp):+.2f}/{np.linalg.norm(Mpp):.2f}")
    print(f"L{l}: "+"\n    ".join(row))
```