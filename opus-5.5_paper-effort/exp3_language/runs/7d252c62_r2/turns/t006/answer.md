```python
import numpy as np
from numpy.linalg import svd, eigvals, norm
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
outs={}
for l in range(3):
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        U,S,Vt=svd(O@V); r=16
        outs[(l,j)]=U[:,:r]
readers={}
for l in range(4):
    w2=W[f'ln2_w.{l}']
    readers[f'fc{l}']=W[f'W_fc.{l}']*w2
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        readers[f'{l}{j}q']=Q; readers[f'{l}{j}k']=K; readers[f'{l}{j}v']=V
def ratio(R,U): return (norm(R@U)**2/norm(R)**2)*(384/U.shape[1])
for (l,j),U in outs.items():
    s=[]
    for nm,R in readers.items():
        lr=int(nm[2]) if nm.startswith('fc') else int(nm[0])
        if nm.startswith('fc'):
            if lr< l: continue
        else:
            if lr<=l: continue
        s.append(f"{nm}:{ratio(R,U):.1f}")
    print(f"L{l}H{j} out read by:", " ".join(s))
```