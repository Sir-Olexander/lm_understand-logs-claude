```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print(np.histogram(W['ln1_w.0'], bins=20))
print(np.sort(W['ln1_w.0'])[:20], np.sort(W['ln1_w.0'])[-20:])
print(np.histogram(W['ln2_w.0'], bins=20))
for l in range(4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        QK=Q.T@K/8; OV=O@V
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        ev=np.linalg.eigvals(OV); 
        print(l,j,'QK fro %.2f top %s'%(np.linalg.norm(QK),sq[:5]), 'OV fro %.2f top %s'%(np.linalg.norm(OV),so[:5]), 'trOV/|..| %.2f'%(ev.real.sum()/np.abs(ev).sum()))
```