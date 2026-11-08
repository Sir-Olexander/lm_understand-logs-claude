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
U,s,_=np.linalg.svd(W['W_proj.3']); U=U[:,:32]
print(np.round(s[:32],1))
rows=[]
for l in [2,3]:
    for j in range(6):
        Q,K,V,O=H[l][j]; OV=O@V
        e=np.linalg.norm(U.T@OV,axis=1); rows.append((f'L{l}H{j}',e/np.linalg.norm(OV)*np.sqrt(384)))
for nm in ['W_proj.0','W_proj.1','W_proj.2']:
    M=W[nm]; e=np.linalg.norm(U.T@M,axis=1); rows.append((nm,e/np.linalg.norm(M)*np.sqrt(384)))
for nm,e in rows: print(f'{nm:9s}', ' '.join(f'{x:4.1f}' for x in e))
```