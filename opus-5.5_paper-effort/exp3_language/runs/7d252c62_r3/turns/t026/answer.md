```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for l in range(4):
    row=[]
    for j in range(6):
        Q,K,V,O=mats(l,j)
        Mk=K@Q.T
        e=np.linalg.eigvals(Mk); sv=np.linalg.svd(Q.T@K,compute_uv=False)[:64]
        ratio=np.sum(np.abs(e))/np.sum(sv)
        tr=np.sum(e.real)/np.sum(sv)
        # OV
        OVs=V@O  # 64x64 equivalent eigen of O@V
        eo=np.linalg.eigvals(OVs); svo=np.linalg.svd(O@V,compute_uv=False)[:64]
        row.append(f"H{j}: QK|e|/sv={ratio:.2f} tr/sv={tr:+.2f}; OV|e|/sv={np.sum(np.abs(eo))/np.sum(svo):.2f} tr/sv={np.sum(eo.real)/np.sum(svo):+.2f}")
    print(f"L{l}: "+" | ".join(row))
```