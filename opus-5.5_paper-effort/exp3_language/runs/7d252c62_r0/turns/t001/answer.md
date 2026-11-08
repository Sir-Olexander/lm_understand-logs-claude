```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    print(k, W[k].shape, float(np.linalg.norm(W[k])), float(W[k].mean()), float(W[k].std()))
d=384; hd=64
print()
for l in range(4):
    lw=W[f'ln1_w.{l}']; lb=W[f'ln1_b.{l}']
    print('layer',l,'ln1_w mean/std/min/max',lw.mean(),lw.std(),lw.min(),lw.max(),' ln1_b norm',np.linalg.norm(lb))
    lw2=W[f'ln2_w.{l}']; lb2=W[f'ln2_b.{l}']
    print('        ln2_w mean/std/min/max',lw2.mean(),lw2.std(),lw2.min(),lw2.max(),' ln2_b norm',np.linalg.norm(lb2))
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        QK=Wq.T@Wk; OV=Wo@Wv
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        qb=Wq@lb; # query bias
        kb_term=np.linalg.norm(Wk.T@qb)
        print(f' L{l}H{j} |QK|={np.linalg.norm(QK):.2f} top sv QK={sq[:5]} |OV|={np.linalg.norm(OV):.3f} top sv OV={so[:4]} |q_bias->k|={kb_term:.3f}')
```