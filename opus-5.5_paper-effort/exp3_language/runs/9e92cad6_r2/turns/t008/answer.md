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
for l in range(4):
    Hl=heads(l)
    for j in range(6):
        QK,OV=Hl[j]
        S=(QK+QK.T)/2; A=(QK-QK.T)/2
        ev=np.linalg.eigvals(QK)
        ev=ev[np.argsort(-np.abs(ev))][:20]
        pos=ev.real.sum()/np.abs(ev).sum()
        ang=np.degrees(np.abs(np.angle(ev)))
        print(f'L{l}H{j} |S|/|A| %.2f  trQK/sum|ev| %.2f  top20 mean|angle| %.0f  top ev'%(np.linalg.norm(S)/np.linalg.norm(A), pos, ang.mean()), np.round(ev[:6],2))
```