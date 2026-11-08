```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q3,K3,V3,O3=mats(0,3); OV3=O3@V3
T=sub(OV3,15,'in')
Q0,K0,V0,O0=mats(0,0); OV0=O0@V0
def ps(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j); QK=Q.T@K/8
        D=T.T@QK@T; I3=T.T@QK@OV3@T; I0=T.T@QK@OV0@T
        nq=np.linalg.norm(QK)
        print('L%dH%d dup |D|/|QK| %.3f pos %.2f diagfrac %.2f | ind3 |I|/|QK| %.3f pos %.2f diag %.2f | ind0 %.3f pos %.2f'%(l,j,np.linalg.norm(D)/nq,ps(D),np.trace(D)/np.abs(D).sum()*15,np.linalg.norm(I3)/nq,ps(I3),np.trace(I3)/np.abs(I3).sum()*15,np.linalg.norm(I0)/nq,ps(I0)))
```