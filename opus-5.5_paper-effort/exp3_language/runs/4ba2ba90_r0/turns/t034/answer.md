```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s_]*g; K=W[f'W_k.{l}'][s_]*g; V=W[f'W_v.{l}'][s_]*g; O=W[f'W_o.{l}'][:,s_]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
Q,K,V,O=H[0][2]; OV02=O@V
u2,s2,vt2=np.linalg.svd(OV02); T=vt2[:20].T
def sc(M):
    m=T.T@M@T; return np.trace(m)/np.linalg.norm(m)
for l in range(1,4):
    for j in range(6):
        Qh,Kh,Vh,Oh=H[l][j]; QK=Qh.T@Kh
        a=sc(QK); b=sc(QK@OV02); c=sc(OV02.T@QK@OV02); d=sc(OV02.T@QK)
        print(f'L{l}H{j} cur-cur {a:6.2f}  cur-prev(induction) {b:6.2f}  prev-prev {c:6.2f}  prev-cur {d:6.2f}   |QK|={np.linalg.norm(QK):.1f}')
```