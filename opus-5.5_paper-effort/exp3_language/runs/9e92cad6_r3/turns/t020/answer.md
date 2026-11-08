```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(1)]
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
U,s,Vt=np.linalg.svd(np.vstack([ov(0,j)/np.linalg.norm(ov(0,j)) for j in [1,2,4,5]]))
Fin=Vt[:2].T
for j in range(6):
    Q,K,V,O=H[0][j]; A=Q.T@K
    M=Fin.T@A@Fin
    print(j,"Fin^T A Fin\n",M, " |A|=",np.linalg.norm(A).round(2), " frac of A on FinxFin", (np.linalg.norm(M)/np.linalg.norm(A)).round(2))
# real Schur of A3 restricted: find its eigenvectors & check Fin overlap with each plane
Q,K,V,O=H[0][3]; A=Q.T@K
ev,evec=np.linalg.eig(A)
idx=np.argsort(-np.abs(ev))[:20]
for i in idx[::2]:
    v=evec[:,i]; plane=np.linalg.qr(np.stack([v.real,v.imag],1))[0]
    print("ev",np.round(ev[i],3),"angle",np.degrees(np.angle(ev[i])).round(1),"overlap with Fin",np.linalg.svd(plane.T@Fin,compute_uv=False).round(2))
```