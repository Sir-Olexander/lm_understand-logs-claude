```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
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
Fout=sub(np.hstack([ov(0,j) for j in [1,2,4,5]]),2,'L')
T=sub(ov(0,3),25,'R'); P=sub(H[0][3][0].T@H[0][3][1],12,'L')
P3out=sub(ov(0,3),25,'L')
print("Fout vs T,P", np.linalg.svd(Fout.T@T,compute_uv=False), np.linalg.svd(Fout.T@P,compute_uv=False))
for l in range(1,4):
    g=W64[f'ln1_w.{l}']; b=W64[f'ln1_b.{l}']
    for j in range(6):
        Q,K,V,O=H[l][j]
        s=slice(64*j,64*j+64)
        qb=W64[f'W_q.{l}'][s]@b
        A=Q.T@K
        # key preference vector from query bias (residual space, applied to xhat_j)
        kp=K.T@qb
        kpn=kp/np.linalg.norm(kp)
        # how much of key-side reading is Fout
        frac_kF=np.linalg.norm(K@Fout)/np.linalg.norm(K)
        # alignment of key preference vector with Fout, P3out
        print(l,j,"|kp|=%.2f"%np.linalg.norm(kp), "kp.Fout", np.round(Fout.T@kpn,2), "kp in P3out %.2f"%np.linalg.norm(P3out.T@kpn),
              "K reads Fout %.2f"%frac_kF, "|A Fout|=%.2f"%np.linalg.norm(A@Fout), "qb.(A Fout) %.2f"%np.linalg.norm(qb@ (K@Fout)))
```