```python
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        out.append((Q,K,V,O))
    return out
H=[heads(l) for l in range(4)]
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l2 in range(1,4):
    for l1 in range(l2):
        print(f"--- reader layer {l2}, writer layer {l1}: rows=reader head; cols writer head: Q-comp | K-comp | V-comp")
        for j2 in range(6):
            Q,K,V,O=H[l2][j2]
            qc=[comp(Q, H[l1][j1][3]@H[l1][j1][2]) for j1 in range(6)]
            kc=[comp(K, H[l1][j1][3]@H[l1][j1][2]) for j1 in range(6)]
            vc=[comp(V, H[l1][j1][3]@H[l1][j1][2]) for j1 in range(6)]
            print(j2, np.array(qc), np.array(kc), np.array(vc))
```