```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W[f'W_q.{l}'][s].astype(np.float64)*g, W[f'W_k.{l}'][s].astype(np.float64)*g, W[f'W_v.{l}'][s].astype(np.float64)*g, W[f'W_o.{l}'][:,s].astype(np.float64)))
    return out
H=[heads(l) for l in range(4)]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return U[:,:k] if side=='L' else Vt[:k].T
Q3,K3,V3,O3=H[0][3]; Q0,K0,V0,O0=H[0][0]
for wh,(Qw,Kw,Vw,Ow) in [(3,H[0][3]),(0,H[0][0])]:
    OV=Ow@Vw
    for k in [10,25]:
        S=sub(OV,k,'R')   # token subspace read by prev head
        print(f"writer L0H{wh}, k={k}")
        for l in range(1,4):
            row=[]
            for j in range(6):
                Q,K,V,O=H[l][j]
                M=S.T@(Q.T@K@OV)@S
                ev=np.linalg.eigvals(M)
                row.append((np.trace(M)/np.abs(ev).sum(), np.linalg.norm(M)))
            print("  L",l, " ".join(f"{a:+.2f}/{b:.1f}" for a,b in row))
```