```python
import numpy as np
from numpy.linalg import svd, eigvals
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for k in sorted(W.keys()):
    a=W[k]
    print(k, a.shape, "std %.4f absmax %.3f fro %.2f"%(a.std(),np.abs(a).max(),np.linalg.norm(a)))
for l in range(4):
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        a=W[f'{nm}.{l}']
        print(l,nm,np.round(np.percentile(a,[0,1,5,25,50,75,95,99,100]),3), 'top dims', np.argsort(-np.abs(a))[:6])
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for l in range(4):
    for j in range(6):
        Q,K,V,O=head(l,j)
        ev=eigvals(K@Q.T)/8
        ev=ev[np.argsort(-np.abs(ev))]
        rf=np.real(ev[:10]).sum()/np.abs(ev[:10]).sum()
        ovs=svd(O@V,compute_uv=False)
        evo=eigvals(V@O); evo=evo[np.argsort(-np.abs(evo))]
        rfo=np.real(evo[:10]).sum()/np.abs(evo[:10]).sum()
        print(f"L{l}H{j} |Q|={np.linalg.norm(Q):.1f} |K|={np.linalg.norm(K):.1f} QKrf={rf:+.2f} QKtop:",
              " ".join(f"{abs(z):.1f}@{np.degrees(np.angle(z)):.0f}" for z in ev[:8]),
              f"| OVsv {ovs[0]:.2f},{ovs[1]:.2f},{ovs[2]:.2f},{ovs[5]:.2f} OVrf={rfo:+.2f}")
```