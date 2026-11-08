```python
for l in range(4):
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]; K=W[f'W_k.{l}'][s]; V=W[f'W_v.{l}'][s]; O=W[f'W_o.{l}'][:,s]
        OV=O@(V*w)
        qb=Q@b  # query bias
        keypref=(K*w).T@qb/8   # score contribution per normalized key vector
        Mqk=(Q*w).T@(K*w)/8
        sv=np.linalg.svd(OV,compute_uv=False)
        print(f"L{l}H{j} |QK|F={np.linalg.norm(Mqk):.3f} |qbias-keypref|={np.linalg.norm(keypref):.3f} |OV|F={np.linalg.norm(OV):.2f} OVsv={np.round(sv[:5],2)} |Wv|={np.linalg.norm(V):.2f} |Wo|={np.linalg.norm(O):.2f}")
```