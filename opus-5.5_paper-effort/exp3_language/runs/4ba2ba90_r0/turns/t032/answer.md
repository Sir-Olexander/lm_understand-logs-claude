```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s_]*g; K=W[f'W_k.{l}'][s_]*g; V=W[f'W_v.{l}'][s_]*g; O=W[f'W_o.{l}'][:,s_]
        res.append((Q,K,V,O))
    return res
H3=heads(3)
U,s,_=np.linalg.svd(W['W_proj.3'])
D=U[:,1:28]
for j in range(6):
    Q,K,V,O=H3[j]; OV=O@V
    M=D.T@OV
    print(f'L3H{j} per-dir gain:', ' '.join(f'{x:.1f}' for x in np.linalg.norm(M,axis=1)))
    print('      sv of D^T OV:', np.round(np.linalg.svd(M,compute_uv=False),1))
```