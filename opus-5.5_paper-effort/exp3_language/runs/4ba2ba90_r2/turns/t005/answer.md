```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
# positional subspace from L0H2 QK
q,k,v,o=head(0,2)
Uq,sq,Vk=np.linalg.svd(q.T@k)
P=np.linalg.qr(np.hstack([Uq[:,:20],Vk[:20].T]))[0]  # combined
Pq=Uq[:,:20]; Pk=Vk[:20].T
print('angle between L0H2 q-side and k-side top20 subspaces: sv', np.linalg.svd(Pq.T@Pk,compute_uv=False)[:20])
info={}
for j in range(6):
    q,k,v,o=head(0,j)
    U,s,Vt=np.linalg.svd(q.T@k)
    Uo,so,Vo=np.linalg.svd(o@v)
    info[j]=(U[:,0],Vt[0],Uo[:,0],Vo[0])
    fr=lambda x: np.linalg.norm(P.T@x)
    print(f'L0H{j}: QK q-dir in P {fr(U[:,0]):.2f}, k-dir in P {fr(Vt[0]):.2f}; OV in-dir in P {fr(Vo[0]):.2f}, OV in-dir2 {fr(Vo[1]):.2f}; out-dir in P {fr(Uo[:,0]):.2f}')
names=['q','k','ovout','ovin']
for t in range(4):
    Mx=np.array([[abs(info[a][t]@info[b][t]) for b in range(6)] for a in range(6)])
    print(names[t]); print(Mx)
```