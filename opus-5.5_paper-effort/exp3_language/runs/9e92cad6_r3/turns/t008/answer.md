```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
for j in range(6):
    Q,K,V,O=H[0][j]
    A=(Q.T@K).astype(np.float64)
    ev=np.linalg.eigvals(A)
    idx=np.argsort(-np.abs(ev))[:12]
    print(j, "sym/asym norm", round(np.linalg.norm(A+A.T)/2,3), round(np.linalg.norm(A-A.T)/2,3))
    print("   |ev|", np.abs(ev[idx]))
    print("   angle deg", np.degrees(np.angle(ev[idx])))
```