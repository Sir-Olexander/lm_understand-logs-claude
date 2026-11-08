```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
Pt=np.eye(384)-T@T.T
Ms={}
for j in range(6):
    Q,K,V,O=mats(0,j); Ms[j]=Pt@(Q.T@K)@Pt
# common pos basis: union of row/col spaces of M1,M5,M2,M3,M0
Z=np.hstack([topright(Ms[1],10),topright(Ms[1].T,10),topright(Ms[5],16),topright(Ms[5].T,16)])
U,S,Vt=np.linalg.svd(Z); print(np.round(S[:40],2))
for d in [12,16,20,24]:
    Bp=U[:,:d]
    m={j:Bp.T@Ms[j]@Bp for j in range(6)}
    kept={j:np.linalg.norm(m[j])/np.linalg.norm(Ms[j]) for j in range(6)}
    def cm(a,b): return np.linalg.norm(a@b-b@a)/(np.linalg.norm(a)*np.linalg.norm(b))
    rng=np.random.default_rng(0)
    base=[]
    for t in range(50):
        Rq,_=np.linalg.qr(rng.normal(size=(d,d)))
        base.append(cm(m[1],Rq@m[5]@Rq.T))
    print(d, 'kept',{j:round(kept[j],2) for j in kept},'comm15',round(cm(m[1],m[5]),3),'rand',round(np.mean(base),3), 'comm with transposes', round(cm(m[1],m[5].T),3))
```