```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
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
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
# MLP-out as writer too? skip
rng=np.random.default_rng(0)
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l2 in range(1,4):
    print('=== layer',l2,'K-comp (rows: later head, cols: earlier head L*H*) ===')
    cols=[(l1,j1) for l1 in range(l2) for j1 in range(6)]
    print('        '+' '.join(f'L{a}H{b}' for a,b in cols))
    for j2 in range(6):
        Wq,Wk,Wv,Wo=H[l2][j2]
        row=[comp(Wk,OV[c]) for c in cols]
        print(f' L{l2}H{j2} K '+' '.join(f'{x:.2f}' for x in row))
    for j2 in range(6):
        Wq,Wk,Wv,Wo=H[l2][j2]
        row=[comp(Wq,OV[c]) for c in cols]
        print(f' L{l2}H{j2} Q '+' '.join(f'{x:.2f}' for x in row))
    for j2 in range(6):
        Wq,Wk,Wv,Wo=H[l2][j2]
        row=[comp(Wv,OV[c]) for c in cols]
        print(f' L{l2}H{j2} V '+' '.join(f'{x:.2f}' for x in row))
# baseline: random gaussian
A=rng.normal(size=(64,384)); B=rng.normal(size=(384,64))@rng.normal(size=(64,384))
print('random baseline',comp(A,B))
```