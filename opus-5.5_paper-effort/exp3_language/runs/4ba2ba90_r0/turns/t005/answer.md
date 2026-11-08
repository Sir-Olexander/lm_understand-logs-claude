```python
def comp_scores():
    out={}
    for l2 in range(1,4):
        for j2 in range(6):
            Q2,K2,V2,O2=H[l2][j2]
            QK2=Q2.T@K2; OV2=O2@V2
            for l1 in range(l2):
                for j1 in range(6):
                    Q1,K1,V1,O1=H[l1][j1]
                    OV1=O1@V1
                    n=np.linalg.norm(OV1)
                    kc=np.linalg.norm(QK2@OV1)/(np.linalg.norm(QK2)*n)
                    qc=np.linalg.norm(OV1.T@QK2)/(np.linalg.norm(QK2)*n)
                    vc=np.linalg.norm(OV2@OV1)/(np.linalg.norm(OV2)*n)
                    out[(l1,j1,l2,j2)]=(qc,kc,vc)
    return out
C=comp_scores()
# baseline random
rng=np.random.default_rng(0)
A=rng.normal(size=(384,64))@rng.normal(size=(64,384)); B=rng.normal(size=(384,64))@rng.normal(size=(64,384))
print('baseline', np.linalg.norm(A@B)/np.linalg.norm(A)/np.linalg.norm(B))
for l2 in range(1,4):
    for j2 in range(6):
        row=[]
        for l1 in range(l2):
            for j1 in range(6):
                q,k,v=C[(l1,j1,l2,j2)]; row.append(f"{l1}{j1}:{q:.2f}/{k:.2f}/{v:.2f}")
        print(f"L{l2}H{j2}", ' '.join(row))
```