```python
def heads(l):
    lw=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append((Q.T@K, O@V))
    return res
H=[heads(l) for l in range(4)]
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
# random baseline
rng=np.random.default_rng(0)
def randlike(M):
    U,s,Vt=np.linalg.svd(M); R1=np.linalg.qr(rng.normal(size=(384,384)))[0]; R2=np.linalg.qr(rng.normal(size=(384,384)))[0]
    return R1@np.diag(s)@R2
for l2 in range(1,4):
    for j2 in range(6):
        QK2,OV2=H[l2][j2]
        rowK=[];rowQ=[];rowV=[]
        for l1 in range(l2):
            for j1 in range(6):
                QK1,OV1=H[l1][j1]
                rowK.append(comp(QK2,OV1)); rowQ.append(comp(QK2.T,OV1)); rowV.append(comp(OV2,OV1))
        print(f'L{l2}H{j2} K:',' '.join('%.2f'%x for x in rowK))
        print(f'      Q:',' '.join('%.2f'%x for x in rowQ))
        print(f'      V:',' '.join('%.2f'%x for x in rowV))
QK2,OV2=H[1][0]; QK1,OV1=H[0][3]
print('baseline', np.mean([comp(randlike(QK2),randlike(OV1)) for _ in range(5)]))
```