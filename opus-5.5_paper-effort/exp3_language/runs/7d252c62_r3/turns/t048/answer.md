```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k], S[:k]
for j in (2,3):
    Q,K,V,O=mats(0,j); U,S=topleft(O@V,6); print("L0H%d OV sv"%j, np.round(S,2))
o3,_=topleft(mats(0,3)[3]@mats(0,3)[2],2); o2,_=topleft(mats(0,2)[3]@mats(0,2)[2],4)
o4,_=topleft(mats(0,4)[3]@mats(0,4)[2],2)
def enrich(M,X,Y):
    return (np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2)/(X.shape[1]*Y.shape[1]/384**2)
subs={'o3':o3,'o2':o2,'o4':o4}
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j); M=Q.T@K
        print(f"L{l}H{j}", " ".join(f"q{a}/k{b}:{enrich(M,subs[a],subs[b]):.0f}" for a in subs for b in subs), "| v o3:", round((np.linalg.norm(V@o3)**2/np.linalg.norm(V)**2)/(2/384),1))
```