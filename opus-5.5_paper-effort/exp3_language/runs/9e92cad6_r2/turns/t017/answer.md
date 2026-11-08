```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q2,K2,V2,O2=mats(0,2); outs=sub(O2@V2,2,'out')
Q3,K3,V3,O3=mats(0,3); T=sub(O3@V3,15,'in'); S3out=sub(O3@V3,15,'out')
for l in range(1,4):
    lb=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q,K,V,O=mats(l,j)
        qb=W[f'W_q.{l}'][s]@lb
        r=K.T@qb/8      # key-only logit vector in normalized-residual space
        # content-content scale: QK frobenius
        QK=Q.T@K/8
        print('L%dH%d |r| %.3f  r in outs %.2f  in T %.2f in S3out %.2f | |QK|fro %.3f top sv %.3f | outs-part of r: %.3f'%(l,j,np.linalg.norm(r),np.linalg.norm(outs.T@r)/np.linalg.norm(r),np.linalg.norm(T.T@r)/np.linalg.norm(r),np.linalg.norm(S3out.T@r)/np.linalg.norm(r),np.linalg.norm(QK),np.linalg.svd(QK,compute_uv=False)[0],np.linalg.norm(outs.T@r)))
```