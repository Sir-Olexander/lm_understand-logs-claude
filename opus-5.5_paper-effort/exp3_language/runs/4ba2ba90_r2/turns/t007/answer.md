```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
q2,k2,v2,o2=head(0,2)
OV2=o2@v2
def evsummary(C,n=12):
    ev=np.linalg.eigvals(C); idx=np.argsort(-np.abs(ev))[:n]; e=ev[idx]
    pos=np.sum(ev.real*np.abs(ev))/np.sum(np.abs(ev)**2)  # weighted positivity
    return e, pos
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=head(l,j)
        C=q.T@k@OV2
        e,pos=evsummary(C)
        # same-token
        e2,pos2=evsummary(q.T@k)
        print(f'L{l}H{j} prevtok-match: posfrac {pos:+.2f} top |ev| {np.abs(e[:4])} args {np.angle(e[:4])}  | QK self posfrac {pos2:+.2f} top {np.abs(e2[:3])} args {np.angle(e2[:3])}')
```