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
A3=H[0][3][0].T@H[0][3][1]
U,s,Vt=np.linalg.svd(A3+A3.T); P=U[:,:14]
A0=H[0][3][0].T@H[0][3][1]
print("L0H3 on P: |M|=%.2f"%np.linalg.norm(P.T@A3@P))
for l in range(0,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K
        M=P.T@A@P
        ev=np.linalg.eigvals(M); i=np.argsort(-np.abs(ev))[:6]
        print(f"L{l}H{j} |M|={np.linalg.norm(M):.2f} sym {np.linalg.norm(M+M.T)/2:.2f} asym {np.linalg.norm(M-M.T)/2:.2f}  ev:",
              " ".join(f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e in ev[i]))
```