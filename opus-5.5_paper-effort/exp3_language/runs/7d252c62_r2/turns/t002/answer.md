```python
import numpy as np
from numpy.linalg import svd, eigvals, norm
np.set_printoptions(linewidth=220, precision=3, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
# (a) QK SVD
for (l,j),(Q,K,V,O) in H.items():
    B=Q.T@K/8
    U,S,Vt=svd(B)
    al=[abs(U[:,i]@Vt[i]) for i in range(5)]
    sym=norm((B+B.T)/2)/norm(B)
    print(f"L{l}H{j} QKsv",np.round(S[:6],3)," |cos(u_i,v_i)|",np.round(al,2), f"sympart {sym:.2f}")
# (b) composition
def rf(M,k=10):
    ev=eigvals(M); ev=ev[np.argsort(-np.abs(ev))][:k]; return np.real(ev).sum()/np.abs(ev).sum()
print("K-comp / Q-comp / V-comp scores (norm ratio) and eig realfrac for Kcomp")
for l2 in range(1,4):
    for j2 in range(6):
        Q2,K2,V2,O2=H[(l2,j2)]
        QK=Q2.T@K2
        s=[]
        for l1 in range(l2):
            for j1 in range(6):
                Q1,K1,V1,O1=H[(l1,j1)]
                OV=O1@V1
                kc=norm(QK@OV)/(norm(QK)*norm(OV)); qc=norm(OV.T@QK)/(norm(QK)*norm(OV))
                vc=norm(V2@OV)/(norm(V2)*norm(OV))
                r=rf((K2@O1)@(V1@Q2.T))
                s.append(f"{l1}.{j1}:k{kc:.2f}q{qc:.2f}v{vc:.2f}r{r:+.1f}")
        print(f"L{l2}H{j2}", " ".join(s))
```