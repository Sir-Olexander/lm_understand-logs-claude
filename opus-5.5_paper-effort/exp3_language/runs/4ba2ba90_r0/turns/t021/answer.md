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
u2,s2,vt2=np.linalg.svd(O@V); T=vt2[:24].T
Uo=np.linalg.svd(W['W_proj.3'])[0][:,:30]
for l in range(4):
  for j in range(6):
    Q,K,V,O=H[l][j]
    u,s,vt=np.linalg.svd(O@V)
    k=20
    a=np.linalg.norm(T.T@vt[:k].T)**2/k; b=np.linalg.norm(Uo.T@u[:,:k])**2/k
    print(f'L{l}H{j} OVin in T: {a:.2f}   OVout in logit-space(Wproj3 top30): {b:.2f}')
```