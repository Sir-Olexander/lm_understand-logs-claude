```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def cos(a,b): return np.sum(a*b)/np.linalg.norm(a)/np.linalg.norm(b)
for l in range(4):
    QK=[];OV=[];Vs=[]
    for j in range(6):
        Q,K,V,O=mats(l,j); QK.append(Q.T@K); OV.append(O@V); Vs.append(V)
    print(f"Layer {l} QK cos:"); print(np.round([[cos(QK[a],QK[b]) for b in range(6)] for a in range(6)],2))
    print(f"Layer {l} OV cos:"); print(np.round([[cos(OV[a],OV[b]) for b in range(6)] for a in range(6)],2))
    # subspace overlap of V input spaces (top 20)
    def tr(M,k=20):
        U,S,Vt=np.linalg.svd(M); return Vt[:k].T
    print(f"Layer {l} OV input-subspace overlap (mean cos^2, top20):")
    print(np.round([[np.mean(np.linalg.svd(tr(OV[a]).T@tr(OV[b]),compute_uv=False)**2) for b in range(6)] for a in range(6)],2))
```