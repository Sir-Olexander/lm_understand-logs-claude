```python
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
for j in range(6):
    q,k,v,o=H[0][j]
    print(f"L0H{j} OV sv", np.round(np.linalg.svd(o@v,compute_uv=False)[:40],3))
    print(f"L0H{j} QK sv", np.round(np.linalg.svd(q.T@k,compute_uv=False)[:40],3))
    print(f"L0H{j} V sv", np.round(np.linalg.svd(v,compute_uv=False)[:40],3))
```