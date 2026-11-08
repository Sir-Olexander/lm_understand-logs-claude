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
comps=[]
for i in range(d):
    x=X[:,i]; y=Y[i]
    l1=y@m[1]@x; l5=y@m[5]@x
    if np.angle(l1)>=0: comps.append((l1,l5))   # choose orientation with M1 angle in [0,pi]
for l1,l5 in comps: print(f"{abs(l1):.2f}@{np.degrees(np.angle(l1)):.0f}  {abs(l5):.2f}@{np.degrees(np.angle(l5)):.0f}")
for hyp,div in [('H1 offset1',1),('H1 offset2',2),('H1 offset3',3)]:
    print(hyp)
    for wt in ['eq','l5']:
        k1=[];k5=[]
        for D in range(0,12):
            s1=s5=0
            for l1,l5 in comps:
                om=np.angle(l1)/div
                a=1.0
                f=1 if abs(np.angle(l1))<1e-6 or abs(abs(np.angle(l1))-np.pi)<1e-6 else 2
                s1+=f*abs(l1)*np.cos(D*om-np.angle(l1))
                s5+=f*abs(l5)*np.cos(D*om-np.angle(l5))
            k1.append(s1);k5.append(s5)
        print(' k1',np.round(k1,1)); print(' k5',np.round(k5,1)); break
```