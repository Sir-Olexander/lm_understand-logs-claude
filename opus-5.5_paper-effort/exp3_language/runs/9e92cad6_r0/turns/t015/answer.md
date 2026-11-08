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
def cos(A,B): return (A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)
for l in range(4):
    QKs=[q.T@k for (q,k,v,o) in H[l]]; OVs=[o@v for (q,k,v,o) in H[l]]
    print("L",l,"QK cos"); print(np.round([[cos(a,b) for b in QKs] for a in QKs],2))
    print("L",l,"OV cos"); print(np.round([[cos(a,b) for b in OVs] for a in OVs],2))
    # subspace overlap of OV outputs (top 16)
    outs=[np.linalg.svd(M)[0][:,:16] for M in OVs]
    print("OV out top16 overlap (mean cos^2)"); print(np.round([[ (np.linalg.svd(a.T@b,compute_uv=False)**2).mean() for b in outs] for a in outs],2))
```