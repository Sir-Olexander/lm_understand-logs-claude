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
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return U[:,:k] if side=='L' else Vt[:k].T
S={'T':sub(ov(0,3),18,'R'),'H3o':sub(ov(0,3),18,'L'),'P':sub(H[0][3][0].T@H[0][3][1]+(H[0][3][0].T@H[0][3][1]).T,12,'L'),
   'Fo':sub(np.hstack([ov(0,j) for j in [1,2,4,5]]),2,'L'),'H0o':sub(ov(0,0),8,'L')}
names=list(S)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K; n=np.linalg.norm(A)
        B=np.array([[np.linalg.norm(S[a].T@A@S[b])/n for b in names] for a in names])
        print(f"L{l}H{j} rows=q {names} cols=k\n", B)
```