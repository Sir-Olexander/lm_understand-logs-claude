```python
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
V={j:W['W_v.0'][64*j:64*j+64] for j in range(6)}
Qr=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
Pp=rs(Qr,30); R=np.eye(384)-Pp@Pp.T
A=rs(V[1]@R,36); B=rs(V[5]@R,36)
U,cs,Vt=np.linalg.svd(A.T@B)
print(np.round(cs[:34],2))
Tsh=A@U[:,:34]   # ordered by sharedness (raw coords)
# MLP0 live neurons reading: raw residual coords; MLP0 input = LN2(resid) - raw directions; Tsh in raw-embedding coords is appropriate for residual
F=W['W_fc.0']*W['ln2_w.0']
live=np.linalg.norm(W['W_fc.0'],axis=1)>1.0
a=F[live]@Tsh
print("MLP0 read norm per shared dir:", np.round(np.linalg.norm(a,axis=0),2))
# random direction baseline
rng=np.random.default_rng(0); Rd=np.linalg.qr(rng.normal(size=(384,20)))[0]
print("random dirs:", np.round(np.linalg.norm(F[live]@Rd,axis=0),2))
```