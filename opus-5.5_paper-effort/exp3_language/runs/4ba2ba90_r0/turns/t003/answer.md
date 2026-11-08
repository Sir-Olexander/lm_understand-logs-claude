```python
d=384; hd=64
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
for l in range(4):
    for j in range(6):
        Q,K,V,O=H[l][j]
        QK=Q.T@K; OV=O@V
        S=(QK+QK.T)/2; A=(QK-QK.T)/2
        sv=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        ev=np.linalg.eigvals(OV); 
        print(f"L{l}H{j} |QK|={np.linalg.norm(QK):7.2f} asym/sym={np.linalg.norm(A)/np.linalg.norm(S):.2f} trQK/|QK|={np.trace(QK)/np.linalg.norm(QK):6.2f} QKsv={sv[:4]} |OV|={np.linalg.norm(OV):6.2f} OVsv={so[:4]} OVeig={ev.real.sum()/np.abs(ev).sum():.2f}")
```