```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:24]]
def tr(Bm):
    s=np.linalg.svd(Bm,compute_uv=False); e=np.linalg.eigvals(Bm); return np.sum(e.real)/np.sum(s), np.linalg.norm(Bm)
L1OV={x:mats(1,x)[3]@mats(1,x)[2] for x in range(6)}
print("Signature: query current token T vs key's token-copied-by-L1Hx (trace/sum|sv| (norm))")
for l in (2,3):
    for j in range(6):
        Q,K,V,O=mats(l,j); M=Q.T@K
        row=[]
        for x in range(6):
            t,n=tr(T.T@M@L1OV[x]@T); row.append(f"{t:+.2f}({n:.2f})")
        # also query prev5 vs key L1Hx-copied T
        row2=[]
        for x in range(6):
            t,n=tr(T.T@OV5.T@M@L1OV[x]@T); row2.append(f"{t:+.2f}({n:.2f})")
        print(f"L{l}H{j} qT/kL1x: "+" ".join(row)+" | q5/kL1x: "+" ".join(row2))
```