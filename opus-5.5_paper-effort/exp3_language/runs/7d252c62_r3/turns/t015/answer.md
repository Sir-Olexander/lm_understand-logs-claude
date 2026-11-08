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
Z=np.hstack([topright(Ms[1],10),topright(Ms[1].T,10),topright(Ms[5],16),topright(Ms[5].T,16)])
U,S,Vt=np.linalg.svd(Z)
d=20; Bp=U[:,:d]
m={j:Bp.T@Ms[j]@Bp for j in range(6)}
C=m[1]/np.linalg.norm(m[1])+0.7*m[5]/np.linalg.norm(m[5])
w_,X=np.linalg.eig(C); Y=np.linalg.inv(X)   # rows of Y are left eigvecs
order=np.argsort(np.angle(w_))
for i in order:
    x=X[:,i]; y=Y[i]
    l1=y@m[1]@x; l5=y@m[5]@x; l2=y@m[2]@x; l3=y@m[3]@x; l0=y@m[0]@x
    nrm=np.linalg.norm(x)*np.linalg.norm(y)
    print(f"{abs(l1)/nrm*0+abs(l1):6.2f}@{np.degrees(np.angle(l1)):5.0f}  M5 {abs(l5):6.2f}@{np.degrees(np.angle(l5)):5.0f}  M2 {abs(l2):5.2f}@{np.degrees(np.angle(l2)):5.0f} M3 {abs(l3):5.2f}@{np.degrees(np.angle(l3)):5.0f}  cond {nrm:.1f}")
```