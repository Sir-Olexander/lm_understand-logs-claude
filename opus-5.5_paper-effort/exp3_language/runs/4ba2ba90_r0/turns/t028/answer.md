```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
srcs={}
for l in range(3):
    srcs[f'A{l}']=np.hstack([H[l][j][3]@H[l][j][2] for j in range(6)])  # all heads OV concatenated (384 x 6*384)
    srcs[f'M{l}']=W[f'W_proj.{l}']
def enr(M,S):
    # expected for random: ||M S||^2 = ||M||^2 ||S||^2 /384
    return (np.linalg.norm(M@S)**2)/(np.linalg.norm(M)**2*np.linalg.norm(S)**2/384)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]
        line=[]
        for nm,M in [('Q',Q),('K',K),('V',V)]:
            line.append(nm+':'+' '.join(f'{s}={enr(M,S):.1f}' for s,S in srcs.items() if int(s[1])<l))
        print(f'L{l}H{j}','  '.join(line))
```