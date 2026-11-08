```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
H={l:heads(l) for l in range(4)}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        QK=Wq.T@Wk
        ev=np.linalg.eigvals(QK)
        idx=np.argsort(-np.abs(ev))[:8]
        sym=(QK+QK.T)/2; asym=(QK-QK.T)/2
        nuc=np.linalg.svd(QK,compute_uv=False).sum()
        print(f'L{l}H{j} tr/nuc={np.trace(QK)/nuc:+.3f} |asym|/|QK|={np.linalg.norm(asym)/np.linalg.norm(QK):.3f} sum|ev|real>0 frac={np.abs(ev[ev.real>0]).sum()/np.abs(ev).sum():.2f}')
        print('    top ev:', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]))
```