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
ev=np.linalg.eigvals(OV2); idx=np.argsort(-np.abs(ev))[:10]
print('OV2 eig', np.abs(ev[idx]), np.angle(ev[idx]))
Uo,so,Vo=np.linalg.svd(OV2)
Uq,sq,Vk=np.linalg.svd(q2.T@k2)
Pq=Uq[:,:20]
print('OV2 in top25 vs P20 overlap sv', np.linalg.svd(Vo[:25]@Pq,compute_uv=False))
print('OV2 out top25 vs in top25 overlap sv', np.linalg.svd(Uo[:,:25].T@Vo[:25].T,compute_uv=False))
# L3H1 details
q,k,v,o=head(3,1)
M=q.T@k
print('L3H1 QK sv', np.linalg.svd(M,compute_uv=False)[:30])
Um,sm,Vm=np.linalg.svd(M)
print('L3H1 QK top10 k-side in OV2 out space(top25):', [round(np.linalg.norm(Uo[:,:25].T@Vm[i]),2) for i in range(10)])
print('L3H1 QK top10 q-side in OV2 out space(top25):', [round(np.linalg.norm(Uo[:,:25].T@Um[:,i]),2) for i in range(10)])
print('L3H1 QK top10 q-side in OV2 in space(top25):', [round(np.linalg.norm(Vo[:25]@Um[:,i]),2) for i in range(10)])
print('L3H1 QK top10 k-side in OV2 in space(top25):', [round(np.linalg.norm(Vo[:25]@Vm[i]),2) for i in range(10)])
```