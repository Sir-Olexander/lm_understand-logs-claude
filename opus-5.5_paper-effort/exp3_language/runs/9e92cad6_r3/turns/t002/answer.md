```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    print("layer",l)
    print(" ln1_w quantiles", np.quantile(W[f'ln1_w.{l}'],[0,.05,.25,.5,.75,.95,1]))
    print(" ln2_w quantiles", np.quantile(W[f'ln2_w.{l}'],[0,.05,.25,.5,.75,.95,1]))
    for j in range(6):
        s=slice(64*j,64*j+64)
        g=W[f'ln1_w.{l}']
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        QK=Q.T@K; OV=O@V
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        # bias contributions
        qb=W[f'W_q.{l}'][s]@W[f'ln1_b.{l}']; kb=W[f'W_k.{l}'][s]@W[f'ln1_b.{l}']
        print(f"  H{j} QK fro {np.linalg.norm(QK):.2f} sv {sq[:6]} | OV fro {np.linalg.norm(OV):.2f} sv {so[:6]} | |qb| {np.linalg.norm(qb):.2f} |kb| {np.linalg.norm(kb):.2f}")
```