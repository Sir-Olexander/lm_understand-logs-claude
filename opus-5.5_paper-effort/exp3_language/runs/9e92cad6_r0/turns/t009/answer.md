```python
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
q,k,v,o=H[0][3]
QK=q.T@k
U,S,Vt=np.linalg.svd(QK)
for r in [4,8,12,16,20,24]:
    ov=np.linalg.svd(U[:,:r].T@Vt[:r].T,compute_uv=False)
    print(r,"overlap of left/right top-r subspaces (sum cos^2 / r):", round((ov**2).sum()/r,3))
# subspace P: union of top 16 left and right
P,_ ,_=np.linalg.svd(np.concatenate([U[:,:16],Vt[:16].T],1),full_matrices=False)
sP=_
print("union sv",np.round(sP,2))
P=P[:,:20]
rng=np.random.default_rng(1)
R=np.linalg.qr(rng.normal(size=(384,20)))[0]
for l in range(4):
    row=[]
    for j in range(6):
        q,k,v,o=H[l][j]; M=q.T@k
        f=np.linalg.norm(P.T@M@P)/np.linalg.norm(M); fr=np.linalg.norm(R.T@M@R)/np.linalg.norm(M)
        fq=np.linalg.norm(P.T@M)/np.linalg.norm(M); fk=np.linalg.norm(M@P)/np.linalg.norm(M)
        row.append(f"H{j}: PP {f:.2f} (rand {fr:.2f}) Pq {fq:.2f} Pk {fk:.2f}")
    print("L",l," | ".join(row))
```