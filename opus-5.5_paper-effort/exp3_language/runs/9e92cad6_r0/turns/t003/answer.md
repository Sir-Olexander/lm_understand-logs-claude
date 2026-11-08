```python
w0=W['ln1_w.0']
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
# eigen structure of QK per head (q^T k), nonzero eigs = eig of k q^T (64x64)
for l in range(4):
    for j in range(6):
        q,k,v,o=H[l][j]
        ev=np.linalg.eigvals(k@q.T)  # eigenvalues of q^T k
        ev=ev[np.argsort(-np.abs(ev))]
        pos=ev.real[ev.real>0].sum(); neg=-ev.real[ev.real<0].sum(); im=np.abs(ev.imag).sum()
        ev2=np.linalg.eigvals(v@o)   # OV eigen (o v)
        ev2=ev2[np.argsort(-np.abs(ev2))]
        print(f"L{l}H{j} QK top eigs {np.round(ev[:5],2)} pos {pos:.2f} neg {neg:.2f} |im| {im:.2f} | OV top eigs {np.round(ev2[:4],2)} posR {ev2.real[ev2.real>0].sum():.2f} negR {-ev2.real[ev2.real<0].sum():.2f}")
```