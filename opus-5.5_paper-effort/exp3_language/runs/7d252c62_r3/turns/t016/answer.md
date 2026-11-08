```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
subs={'T':T}
subs['o0.1']=np.linalg.qr(O1@V1@T)[0]; subs['o0.5']=np.linalg.qr(O5@V5@T)[0]
subs['o0.4']=topleft(mats(0,4)[3]@mats(0,4)[2],2)
for j in range(6):
    Q,K,V,O=mats(1,j); subs[f'o1.{j}']=topleft(O@V,16)
for j in range(6):
    Q,K,V,O=mats(2,j); subs[f'o2.{j}']=topleft(O@V,16)
print("overlap o0.1 vs o0.5", np.round(np.linalg.svd(subs['o0.1'].T@subs['o0.5'],compute_uv=False),2))
def fr(Mat,X):  # fraction of row-space energy of Mat in subspace X, divided by random expectation
    return (np.linalg.norm(Mat@X)**2/np.linalg.norm(Mat)**2)/(X.shape[1]/384)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j)
        out=[]
        for nm,X in subs.items():
            if nm.startswith(f'o{l}') or (nm.startswith('o2') and l<3) : continue
            out.append(f"{nm}:q{fr(Q,X):.1f}/k{fr(K,X):.1f}/v{fr(V,X):.1f}")
        print(f"L{l}H{j}", " ".join(out))
```