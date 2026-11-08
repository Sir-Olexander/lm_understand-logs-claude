```python
def rowspace(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T, S
w=W['ln1_w.0']
V1=W['W_v.0'][64:128]; V5=W['W_v.0'][320:384]; V2=W['W_v.0'][128:192]
for scale in ['raw','w']:
    a=V1 if scale=='raw' else V1*w; b=V5 if scale=='raw' else V5*w; c=V2 if scale=='raw' else V2*w
    for k in (24,28,32,36,40):
        A,_=rowspace(a,k); B,_=rowspace(b,k)
        cs=np.linalg.svd(A.T@B,compute_uv=False)
        print(scale,k, "cos:", np.round(cs,2))
```