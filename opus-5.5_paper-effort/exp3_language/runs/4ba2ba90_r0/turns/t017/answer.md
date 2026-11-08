```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
Q,K,V,O=H[0][2]
OV02=O@V
u2,s2,vt2=np.linalg.svd(OV02); T=vt2[:20].T
def score(M):
    m=T.T@M@T; return np.trace(m)/np.linalg.norm(m), np.linalg.norm(m)/np.linalg.norm(M)
for l in range(1,4):
    for j in range(6):
        Qh,Kh,Vh,Oh=H[l][j]
        QK=Qh.T@Kh
        a=score(QK@OV02)      # induction: query current token, key prev token via L0H2
        b=score(QK)           # same-token matching
        c=score(OV02.T@QK)    # reverse
        print(f'L{l}H{j} induction tr/F={a[0]:6.2f} (mass {a[1]:.3f})  same-token tr/F={b[0]:6.2f}  rev={c[0]:6.2f}')
```