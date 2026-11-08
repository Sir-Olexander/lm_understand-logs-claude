```python
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def cs_sub(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
V={j:W['W_v.0'][64*j:64*j+64] for j in range(6)}
Qr=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
Pp=rs(Qr,30); R=np.eye(384)-Pp@Pp.T
for (a,b) in [(1,5),(1,2),(5,2)]:
    for k in (32,40):
        c=cs_sub(rs(V[a]@R,k),rs(V[b]@R,k))
        print(a,b,k, np.round(c[18:34],2))
```