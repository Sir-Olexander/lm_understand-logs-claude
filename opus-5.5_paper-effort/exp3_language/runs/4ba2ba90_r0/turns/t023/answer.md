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
QK=Q.T@K
u,s,vt=np.linalg.svd(QK)
P=np.linalg.svd(np.hstack([u[:,:22],vt[:22].T]),full_matrices=False)[0][:,:24]
for l in range(4):
    for j in range(6):
        Qh,Kh,Vh,Oh=H[l][j]; QKh=Qh.T@Kh
        Bp=P.T@QKh@P
        ev=np.linalg.eigvals(Bp); o=np.argsort(-np.abs(ev))[:8]
        frac=np.linalg.norm(Bp)/np.linalg.norm(QKh)
        print(f'L{l}H{j} |Bp|={np.linalg.norm(Bp):6.2f} frac={frac:.2f} top eig:', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[o]))
```