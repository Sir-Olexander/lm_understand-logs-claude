```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
OVb=np.hstack([head(0,j)[3]@head(0,j)[2] for j in [0,1,4,5]])
D=np.linalg.svd(OVb)[0][:,:3]
U0=np.linalg.svd(W['W_proj.3'])[0][:,0]
# common-written directions: top left sv of concatenated writers
Wr=np.hstack([W[f'W_o.{l}'] for l in range(4)]+[W[f'W_proj.{l}'] for l in range(4)])
Uw,sw,_=np.linalg.svd(Wr); print('all-writers top sv', sw[:6]); 
print('U0 vs Uw0', abs(U0@Uw[:,0]))
for l in [1,2,3]:
    b=W[f'ln1_b.{l}']; g=W[f'ln1_w.{l}']
    print(f'layer {l}: |ln1_b|=%.2f, cos(ln1_b, U0)=%.2f'%(np.linalg.norm(b), (b@U0)/np.linalg.norm(b)))
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][sl]; Wk=W[f'W_k.{l}'][sl]
        qb=Wq@b                     # constant query
        kpref=(Wk*g).T@qb           # key preference direction (on xhat)
        q,k,v,o=head(l,j); M=q.T@k
        Z=M@D; 
        print(f' L{l}H{j} |qb|=%.2f |kpref|=%.2f  kpref in D: %.2f  | query partners of D: cos with U0 %s, with ln1_b-ish? ; |q| dir of U0 read: %.2f'%(np.linalg.norm(qb),np.linalg.norm(kpref),np.linalg.norm(D.T@kpref)/np.linalg.norm(kpref), np.round([abs(U0@Z[:,i])/np.linalg.norm(Z[:,i]) for i in range(2)],2), np.linalg.norm(q@U0)))
```