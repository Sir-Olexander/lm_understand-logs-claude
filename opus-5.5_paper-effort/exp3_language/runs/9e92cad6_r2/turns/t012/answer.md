```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def heads(l):
    lw=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append((Q.T@K/8, O@V))
    return res
H=[heads(l) for l in range(4)]
def posscore(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
for l2 in range(1,4):
    for j2 in range(6):
        QK2=H[l2][j2][0]
        row=[]
        for l1 in range(l2):
            for j1 in range(6):
                row.append(posscore(QK2@H[l1][j1][1]))
        print(f'L{l2}H{j2}',' '.join('%+.2f'%x for x in row))
```