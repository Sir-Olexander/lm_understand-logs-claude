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
w_,X=np.linalg.eig(C); Y=np.linalg.inv(X)
Q4,K4,V4,O4=mats(0,4); in4=topright(O4@V4,2)
Q0,K0,V0,O0=mats(0,0); in0=topright(O0@V0,2)
Q3,K3,V3,O3=mats(0,3); in3=topright(O3@V3,1)
print("H4 input dirs norm in Bp:", np.round(np.linalg.norm(Bp.T@in4,axis=0),2), "H0:",np.round(np.linalg.norm(Bp.T@in0,axis=0),2), "H3:",np.round(np.linalg.norm(Bp.T@in3,axis=0),2))
order=np.argsort(np.angle(w_))
for i in order:
    x=X[:,i]; y=Y[i]
    l1=y@m[1]@x; l5=y@m[5]@x
    if np.angle(l1)<-1e-6: continue
    e4=[abs(y@(Bp.T@in4[:,r]))*np.linalg.norm(x) for r in range(2)]
    e0=[abs(y@(Bp.T@in0[:,r]))*np.linalg.norm(x) for r in range(2)]
    e3=abs(y@(Bp.T@in3[:,0]))*np.linalg.norm(x)
    print(f"M1 {abs(l1):.2f}@{np.degrees(np.angle(l1)):4.0f} M5 {abs(l5):.2f}@{np.degrees(np.angle(l5)):4.0f} | H4in {e4[0]:.2f},{e4[1]:.2f} H0in {e0[0]:.2f},{e0[1]:.2f} H3in {e3:.2f}")
```