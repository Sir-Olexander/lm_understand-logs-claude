```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l2 in [1,2,3]:
    print(f'== layer {l2}: K-comp (rows: src head L0..L{l2-1}, cols: dst heads)')
    for l1 in range(l2):
        for a in range(6):
            q,k,v,o=H[(l1,a)]; OV=o@v
            kc=[comp(H[(l2,b)][1],OV) for b in range(6)]
            qc=[comp(H[(l2,b)][0],OV) for b in range(6)]
            vc=[comp(H[(l2,b)][2],OV) for b in range(6)]
            print(f' L{l1}H{a}  K',np.array(kc),' Q',np.array(qc),' V',np.array(vc))
```