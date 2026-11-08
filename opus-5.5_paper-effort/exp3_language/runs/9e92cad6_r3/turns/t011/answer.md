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
T=sub(ov(0,3),25,'R'); P=sub(H[0][3][0].T@H[0][3][1],12,'L'); Pk=sub(H[0][3][0].T@H[0][3][1],12,'R')
Fin=sub(np.vstack([ov(0,j) for j in [1,2,4,5]]),2,'R')
b0=W64['ln1_b.0']
for j in range(6):
    Q,K,V,O=H[0][j]
    s=slice(64*j,64*j+64)
    qb=W64['W_q.0'][s]@b0
    kp=K.T@qb
    A=Q.T@K
    U,sv,Vt=np.linalg.svd(A)
    qd=U[:,0]; kd=Vt[0]
    print(j,"sv1 %.2f"%sv[0],"qd in T %.2f P %.2f Fin %.2f"%(np.linalg.norm(T.T@qd),np.linalg.norm(P.T@qd),np.linalg.norm(Fin.T@qd)),
          "| kd in T %.2f Pk %.2f Fin %.2f"%(np.linalg.norm(T.T@kd),np.linalg.norm(Pk.T@kd),np.linalg.norm(Fin.T@kd)),
          "| |kp| %.2f kp in T %.2f Pk %.2f Fin %.2f"%(np.linalg.norm(kp),np.linalg.norm(T.T@kp)/np.linalg.norm(kp),np.linalg.norm(Pk.T@kp)/np.linalg.norm(kp),np.linalg.norm(Fin.T@kp)/np.linalg.norm(kp)),
          "kd.kp cos %.2f"%(kd@kp/np.linalg.norm(kp)))
```