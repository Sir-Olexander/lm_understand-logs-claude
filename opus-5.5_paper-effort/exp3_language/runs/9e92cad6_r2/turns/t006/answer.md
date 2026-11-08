```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append((Q.T@K/8, O@V))
    return res
H=[heads(l) for l in range(4)]
def topvecs(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return (U[:,:k] if side=='out' else Vt[:k].T), s[:k]
outs={}; ins={}
for j in range(6):
    QK,OV=H[0][j]
    outs[j]=topvecs(OV,2,'out')[0]; ins[j]=topvecs(OV,2,'in')[0]
print('overlap of OV output 2D subspaces (L0):')
for a in range(6):
    print(a,[np.round(np.linalg.svd(outs[a].T@outs[b],compute_uv=False),2) for b in range(6)])
print('overlap of OV input 2D subspaces (L0):')
for a in range(6):
    print(a,[np.round(np.linalg.svd(ins[a].T@ins[b],compute_uv=False),2) for b in range(6)])
b=W['ln1_b.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    qb=W['W_q.0'][s]@b; kb=W['W_k.0'][s]@b
    Q=W['W_q.0'][s]*W['ln1_w.0']; K=W['W_k.0'][s]*W['ln1_w.0']
    keyterm=qb@K/8; qterm=kb@Q/8
    print(j,'|qb| %.3f |kb| %.3f  key-only vec norm %.3f, query-only %.3f'%(np.linalg.norm(qb),np.linalg.norm(kb),np.linalg.norm(keyterm),np.linalg.norm(qterm)))
```