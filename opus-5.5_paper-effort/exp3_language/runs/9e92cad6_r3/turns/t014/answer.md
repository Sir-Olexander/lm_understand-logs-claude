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
T=sub(ov(0,3),18,'R')
OV3=ov(0,3)
def pos(M):
    ev=np.linalg.eigvals(M); return np.real(np.trace(M))/np.abs(ev).sum()
print("head    same(T,T)      induction(T,OV3T)     prevprev(OV3T,OV3T)   reverse(OV3T,T)")
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K
        Ms=T.T@A@T; Mi=T.T@A@OV3@T; Mp=T.T@OV3.T@A@OV3@T; Mr=T.T@OV3.T@A@T
        print(f"L{l}H{j}", " | ".join(f"{pos(M):+.2f} n{np.linalg.norm(M):.2f}" for M in [Ms,Mi,Mp,Mr]))
```