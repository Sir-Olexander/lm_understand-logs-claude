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
rng=np.random.default_rng(0)
A=rng.normal(size=(384,64))@rng.normal(size=(64,384)); B=rng.normal(size=(384,64))@rng.normal(size=(64,384))
print('baseline', np.linalg.norm(A@B)/np.linalg.norm(A)/np.linalg.norm(B))
for l2 in range(1,4):
    for j2 in range(6):
        Q2,K2,V2,O2=H[l2][j2]; QK2=Q2.T@K2; OV2=O2@V2
        row=[]
        for l1 in range(l2):
            for j1 in range(6):
                Q1,K1,V1,O1=H[l1][j1]; OV1=O1@V1; n=np.linalg.norm(OV1)
                kc=np.linalg.norm(QK2@OV1)/(np.linalg.norm(QK2)*n)
                qc=np.linalg.norm(OV1.T@QK2)/(np.linalg.norm(QK2)*n)
                vc=np.linalg.norm(OV2@OV1)/(np.linalg.norm(OV2)*n)
                row.append(f"{l1}{j1}:{qc:.2f}/{kc:.2f}/{vc:.2f}")
        print(f"L{l2}H{j2}", ' '.join(row))
```