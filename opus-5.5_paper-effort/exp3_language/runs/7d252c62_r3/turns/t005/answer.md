```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    Q=W[f'W_q.{l}'][s]*w; K=W[f'W_k.{l}'][s]*w; V=W[f'W_v.{l}'][s]*w; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
res={}
for l2 in range(1,4):
    for j2 in range(6):
        Q2,K2,V2,O2=mats(l2,j2)
        QK=Q2.T@K2; OV2=O2@V2
        line=[]
        for l1 in range(l2):
            for j1 in range(6):
                Q1,K1,V1,O1=mats(l1,j1)
                OV1=O1@V1
                kc=comp(QK,OV1); qc=comp(OV1.T,QK); vc=comp(OV2,OV1)
                line.append(f"{l1}.{j1}:K{kc:.2f}/Q{qc:.2f}/V{vc:.2f}")
        print(f"L{l2}H{j2}", " ".join(line))
# baseline random
rng=np.random.default_rng(0)
A=rng.normal(size=(384,64))@rng.normal(size=(64,384)); B=rng.normal(size=(384,64))@rng.normal(size=(64,384))
print("random baseline", comp(A,B))
```