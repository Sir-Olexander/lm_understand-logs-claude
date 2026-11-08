```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
for j in [3,0]:
    Q,K,V,O=mats(0,j); QK=Q.T@K/8
    ev=np.linalg.eigvals(QK); ev=ev[np.argsort(-np.abs(ev))]
    print('L0H%d'%j)
    for e in ev[:50]:
        if e.imag>=0: print('  |%.3f| ang %.0f'%(abs(e),np.degrees(np.angle(e))),end='')
    print()
```