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
U=np.linalg.svd(np.hstack([u[:,:26],vt[:26].T]),full_matrices=False)[0][:,:28]
B=U.T@QK@U
ev=np.linalg.eigvals(B)
o=np.argsort(-np.abs(ev))
for e in ev[o]: print(np.round(abs(e),2), np.round(np.degrees(np.angle(e)),1))
```