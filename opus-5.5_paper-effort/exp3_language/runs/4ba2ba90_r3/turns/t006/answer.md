```python
import numpy as np
np.set_printoptions(linewidth=200, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A=q.T@k
U,s,Vt=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U[:,:18],Vt[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v)
T=Vt2[:24].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
OVp=o@v   # prev-token head OV
def frac(M,P,side):  # fraction of Frobenius of bilinear M on subspace P
    return np.linalg.norm(P.T@M if side=='q' else M@P)/np.linalg.norm(M)
def posr(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
print("baseline frac for random 24d: %.2f"%np.sqrt(24/384))
for l in range(4):
    for h in range(6):
        q,k,v,o=head(l,h); A=q.T@k; OV=o@v
        line=f"L{l}H{h}: QK q-on-pos {frac(A,Ppos,'q'):.2f} k-on-pos {frac(A,Ppos,'k'):.2f} q-on-tok {frac(A,T,'q'):.2f} k-on-tok {frac(A,T,'k'):.2f} | OVin-on-pos {np.linalg.norm(OV@Ppos)/np.linalg.norm(OV):.2f} OVin-on-tok {np.linalg.norm(OV@T)/np.linalg.norm(OV):.2f}"
        if l>0:
            Mtt=T.T@A@T; Mind=T.T@A@OVp@T; Mpp=Ppos.T@A@Ppos
            line+=f" | same-tok {posr(Mtt):+.2f}({np.linalg.norm(Mtt):.1f}) induct {posr(Mind):+.2f}({np.linalg.norm(Mind):.1f}) pospos({np.linalg.norm(Mpp):.1f})"
        print(line)
```