```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return U[:,:k] if side=='L' else Vt[:k].T
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
Fo=sub(np.hstack([ov(0,j) for j in [1,2,4,5]]),2,'L')
Fi=sub(np.vstack([ov(0,j) for j in [1,2,4,5]]),2,'R')
for j in [1,2,4,5]:
    print("L0H",j,"Fo^T OV Fi:\n",Fo.T@ov(0,j)@Fi)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K
        s=slice(64*j,64*j+64); qb=W64[f'W_q.{l}'][s]@W64[f'ln1_b.{l}']
        print(f"L{l}H{j} FoAFo", (Fo.T@A@Fo).ravel(), " qb-term", (qb@(K@Fo))/1, " ln1_w on Fo", np.round(np.sqrt(((Fo**2).T@W64[f'ln1_w.{l}']**2)),2), "ln1_b.Fo", np.round(Fo.T@W64[f'ln1_b.{l}'],2))
```