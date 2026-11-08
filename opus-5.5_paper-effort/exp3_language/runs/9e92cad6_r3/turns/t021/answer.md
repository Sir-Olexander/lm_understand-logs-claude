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
U,s,Vt=np.linalg.svd(ov(0,3))
T=Vt[:28].T
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K
        M=T.T@A@T; Ms=(M+M.T)/2
        e=np.sort(np.linalg.eigvalsh(Ms))
        print(f"L{l}H{j}", e[:10], "...", e[-6:])
```