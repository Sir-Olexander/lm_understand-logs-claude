```python
def qk_eigs(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    Q=W[f'W_q.{l}'][s]*w; K=W[f'W_k.{l}'][s]*w
    M=K@Q.T   # 64x64; eigenvalues equal those of Q^T K
    e=np.linalg.eigvals(M)
    e=e[np.argsort(-np.abs(e))]
    return e
for l in range(4):
    for j in range(6):
        e=qk_eigs(l,j)
        top=e[:10]
        re=np.sum(e.real)/np.sum(np.abs(e))
        print(f"L{l}H{j} sum|e|={np.sum(np.abs(e)):.1f} realfrac={re:.2f} top:", " ".join(f"{abs(z):.1f}@{np.degrees(np.angle(z)):.0f}" for z in top))
```