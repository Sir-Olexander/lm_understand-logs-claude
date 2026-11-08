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
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:28]]
F=W['W_fc.0']*W['ln2_w.0']
live=np.linalg.norm(W['W_fc.0'],axis=1)>1.0
F=F[live]; print("live",live.sum())
# decompose each live neuron's input into subspaces
Ppos_list=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Ppos_list.append(topright(Q,30)); Ppos_list.append(topright(K,30))
Pp=sum(x@x.T for x in Ppos_list)/4
e2,v2=np.linalg.eigh(Pp); i2=np.argsort(-e2); Ppos=v2[:,i2[:24]]; Ppos=np.linalg.qr(Ppos-T@(T.T@Ppos))[0]
o5=np.linalg.qr(OV5@T)[0][:,:20]; o1=np.linalg.qr(OV1@T)[0][:,:20]
o4=np.linalg.svd(mats(0,4)[3]@mats(0,4)[2])[0][:,:2]
tot=np.linalg.norm(F,axis=1)**2
for nm,X in [('T',T),('pos',Ppos),('o5',o5),('o1',o1),('o4',o4)]:
    fr=np.linalg.norm(F@X,axis=1)**2/tot
    print(nm, X.shape[1], "mean frac", fr.mean().round(3), "quantiles", np.round(np.quantile(fr,[.1,.5,.9]),3))
```