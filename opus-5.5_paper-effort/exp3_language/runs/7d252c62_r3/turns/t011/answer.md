```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
T=np.load('/tmp/T.npy') if False else None
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
# positional subspace guess: input subspaces of QK of H1,H5 (q and k sides)
Qs=[];
for j in (1,5):
    Q,K,V,O=mats(0,j)
    Qs.append(topright(Q,30)); Qs.append(topright(K,30))
Pp=sum(x@x.T for x in Qs)/4
ev2,evec2=np.linalg.eigh(Pp); idx2=np.argsort(-ev2); print("pos-subspace eig",np.round(ev2[idx2[:50]],2))
Ppos=evec2[:,idx2[:30]]
for j in (0,4,2,3):
    Q,K,V,O=mats(0,j)
    U,S,Vt=np.linalg.svd(O@V)
    for r in range(3):
        v=Vt[r]
        print(f"H{j} OV in-dir {r} sv {S[r]:.2f}: in T {np.linalg.norm(T.T@v):.2f}, in Ppos {np.linalg.norm(Ppos.T@v):.2f}, cos with ln1_b.0-dir {abs(v@W['ln1_b.0'])/np.linalg.norm(W['ln1_b.0']):.2f}")
# overlap of H4 and H0 output dirs
U4=topleft(mats(0,4)[3]@mats(0,4)[2],2); U0=topleft(mats(0,0)[3]@mats(0,0)[2],2)
print("out overlap H4,H0", np.round(np.linalg.svd(U4.T@U0,compute_uv=False),3))
V4=topright(mats(0,4)[3]@mats(0,4)[2],2); V0=topright(mats(0,0)[3]@mats(0,0)[2],2)
print("in overlap H4,H0", np.round(np.linalg.svd(V4.T@V0,compute_uv=False),3))
```