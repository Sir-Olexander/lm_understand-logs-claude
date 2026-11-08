```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
q2,k2,v2,o2=head(0,2)
Uq,sq,Vk=np.linalg.svd(q2.T@k2)
P=np.linalg.svd(np.hstack([Uq[:,:26],Vk[:26].T]),full_matrices=False)[0][:,:26]
Pc=np.eye(384)-P@P.T
_,st,Vt=np.linalg.svd((o2@v2)@Pc)
T=Vt[:24].T  # token-ish subspace
print('T vs P overlap', np.linalg.norm(T.T@P))
# also input subspace of all L0: stack all reading weights
R=np.vstack([W['W_q.0']*W['ln1_w.0'],W['W_k.0']*W['ln1_w.0'],W['W_v.0']*W['ln1_w.0']])
_,sR,VR=np.linalg.svd(R)
print('stacked L0 attn reading sv', sR[:80])
for j in [0,1,3,4,5]:
    q,k,v,o=head(0,j)
    U,s,Vh=np.linalg.svd(q.T@k)
    for nm,x in [('qdir',U[:,0]),('kdir',Vh[0]),('qdir2',U[:,1]),('kdir2',Vh[1])]:
        print(f'L0H{j} {nm}: |P| %.2f |T| %.2f  rest %.2f'%(np.linalg.norm(P.T@x),np.linalg.norm(T.T@x),np.sqrt(max(0,1-np.linalg.norm(P.T@x)**2-np.linalg.norm(T.T@x)**2))))
    Uo,so,Vo=np.linalg.svd(o@v)
    print(f'   OV in-dir1: |P| %.2f |T| %.2f ; in-dir2 |P| %.2f |T| %.2f'%(np.linalg.norm(P.T@Vo[0]),np.linalg.norm(T.T@Vo[0]),np.linalg.norm(P.T@Vo[1]),np.linalg.norm(T.T@Vo[1])))
```