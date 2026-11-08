```python
rng=np.random.default_rng(1)
V1=W['W_v.0'][64:128]; V5=W['W_v.0'][320:384]; V2=W['W_v.0'][128:192]
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
w=W['ln1_w.0']
# positional subspace (raw coords) via Q/K of heads 1,5 raw rows
Qr=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
for npos in (0,10,20,30):
    if npos>0:
        Pp=rs(Qr,npos); R=np.eye(384)-Pp@Pp.T
    else: R=np.eye(384)
    out=[]
    for k in (32,40):
        A=rs(V1@R,k); B=rs(V5@R,k)
        cs=np.linalg.svd(A.T@B,compute_uv=False)
        out.append(np.round(cs[16:34],2))
    print("npos",npos); [print("  ",x) for x in out]
# also three-way: V1,V5,V2
A=rs(V1,40); B=rs(V5,40); C=rs(V2,40)
U,cs,Vt=np.linalg.svd(A.T@B); S=A@U[:,:30]
print("shared(1,5) vs V2 cos", np.round(np.linalg.svd(S.T@C,compute_uv=False),2))
```