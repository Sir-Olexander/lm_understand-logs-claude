```python
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def cs_sub(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
V={j:W['W_v.0'][64*j:64*j+64] for j in range(6)}
for npos in (30,45,60,80):
    Qr=np.vstack([W['W_q.0'][64*j:64*j+64] for j in (0,1,3,4,5)]+[W['W_k.0'][64*j:64*j+64] for j in (0,1,3,4,5)]+[V[4],V[0],V[3]])
    Pp=rs(Qr,npos); R=np.eye(384)-Pp@Pp.T
    c=cs_sub(rs(V[1]@R,36),rs(V[5]@R,36))
    print(npos, np.round(c[16:34],2))
# random baseline for max cos with 36-d in (384-npos) dims
rng=np.random.default_rng(0)
for n in (354,304):
    A=np.linalg.qr(rng.normal(size=(n,36)))[0];B=np.linalg.qr(rng.normal(size=(n,36)))[0]
    print("rand",n,np.round(cs_sub(A,B)[:3],2))
```