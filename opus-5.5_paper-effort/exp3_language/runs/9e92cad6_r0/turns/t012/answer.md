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
OV3=H[0][3][3]@H[0][3][2]
OV0=H[0][0][3]@H[0][0][2]
U3,S3,Vt3=np.linalg.svd(OV3); T=Vt3[:16].T
rng=np.random.default_rng(0)
for src,OV in [("L0H3",OV3),("L0H0",OV0)]:
    print("source",src)
    for l in [1,2,3]:
        row=[]
        for j in range(6):
            q,k,v,o=H[l][j]; QK=q.T@k
            M=T.T@QK@OV@T
            ev=np.linalg.eigvals(M)
            row.append(f"H{j} tr/F {np.trace(M)/np.linalg.norm(M):+.2f} |M| {np.linalg.norm(M):.2f} maxRe {ev.real.max():.2f} minRe {ev.real.min():.2f}")
        print(" L",l," | ".join(row))
# direct token-token QK within T for each later head (same-token attention?)
print("direct T-T QK")
for l in range(4):
    row=[]
    for j in range(6):
        q,k,v,o=H[l][j]; M=T.T@(q.T@k)@T
        row.append(f"H{j} tr/F {np.trace(M)/np.linalg.norm(M):+.2f} |M| {np.linalg.norm(M):.2f}")
    print(" L",l," | ".join(row))
```