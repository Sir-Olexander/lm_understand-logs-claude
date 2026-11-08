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
def stat(QK,OV):
    M=QK@OV
    ev=np.linalg.eigvals(M)
    return np.trace(M)/np.linalg.norm(M), ev.real[ev.real>0].sum()/(np.abs(ev.real).sum()+1e-9), np.linalg.norm(M)/(np.linalg.norm(QK)*np.linalg.norm(OV))
for lk in [2,3]:
    for j in range(6):
        q,k,v,o=H[lk][j]; QK=q.T@k
        row=[]
        for ls in range(lk):
            for g in range(6):
                qq,kk,vv,oo=H[ls][g]
                tr,fp,kc=stat(QK,oo@vv)
                row.append(f"{tr:+.1f}")
        print(f"L{lk}H{j} trace/F with sources L0..L{lk-1} heads:", " ".join(row))
# MLP-mediated: QK of L3 heads with W_proj.l @ W_fc.l (linearized, no relu) is meaningless; skip
```