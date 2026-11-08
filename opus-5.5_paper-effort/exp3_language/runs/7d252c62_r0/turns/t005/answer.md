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
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    s=np.linalg.svd(Wo@Wv,compute_uv=False)
    print(f'L0H{j} OV sv:',s[:40].round(3))
    s=np.linalg.svd(Wv,compute_uv=False); print('      Wv sv:',s[:40].round(3))
    s=np.linalg.svd(Wk,compute_uv=False); print('      Wk sv:',s[:40].round(3))
    s=np.linalg.svd(Wq,compute_uv=False); print('      Wq sv:',s[:40].round(3))
```