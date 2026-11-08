```python
for l in [0]:
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        QK=Q.T@K/8
        S=(QK+QK.T)/2; A=(QK-QK.T)/2
        ev=np.linalg.eigvals(QK)
        idx=np.argsort(-np.abs(ev))[:12]
        print(j,'|S| %.3f |A| %.3f'%(np.linalg.norm(S),np.linalg.norm(A)))
        print('  ev',np.round(ev[idx],3))
        # subspace overlap between Q row space and K row space
        uq=np.linalg.svd(Q,full_matrices=False)[2][:8]; uk=np.linalg.svd(K,full_matrices=False)[2][:8]
        print('  Q-K top8 subspace overlap svals', np.round(np.linalg.svd(uq@uk.T,compute_uv=False),2))
        print('  Q svals', np.round(np.linalg.svd(Q,compute_uv=False)[:8],2), 'K svals', np.round(np.linalg.svd(K,compute_uv=False)[:8],2))
```