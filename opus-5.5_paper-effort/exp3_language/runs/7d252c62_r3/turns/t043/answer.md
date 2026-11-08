```python
rng=np.random.default_rng(1)
V1=W['W_v.0'][64:128]; V5=W['W_v.0'][320:384]; V2=W['W_v.0'][128:192]
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
for k in (28,32,40):
    A=rs(V1,k); B=rs(V5,k)
    cs=np.linalg.svd(A.T@B,compute_uv=False)
    # random baseline
    Ar=np.linalg.qr(rng.normal(size=(384,k)))[0]; Br=np.linalg.qr(rng.normal(size=(384,k)))[0]
    cr=np.linalg.svd(Ar.T@Br,compute_uv=False)
    print(k,"count cos>rand max", np.sum(cs>cr[0]), "rand max", cr[0].round(2))
# simulate: learned V = random 64xN coefficient * token basis with decaying strengths + init noise
def sim(N,strengths,noise=0.017):
    Bt=np.linalg.qr(rng.normal(size=(384,N)))[0]
    def mk():
        C=rng.normal(size=(64,N))*strengths[None,:]
        return C@Bt.T + rng.normal(0,noise,(64,384))
    a=mk(); b=mk()
    out=[]
    for k in (28,32,40):
        cs=np.linalg.svd(rs(a,k).T@rs(b,k),compute_uv=False); out.append(np.round(cs[18:32],2))
    return out, np.round(np.linalg.svd(a,compute_uv=False)[:34],2)
# calibrate strength so top sv ~2.2: C entries std s gives sv ~ s*sqrt(64)
for N in (24,28):
    st=np.linspace(0.3,0.1,N)
    o,sv=sim(N,st)
    print("N",N,"sv",sv)
    for x in o: print("   ",x)
print("real V1 sv", np.round(np.linalg.svd(V1,compute_uv=False)[:34],2))
print("real V5 sv", np.round(np.linalg.svd(V5,compute_uv=False)[:34],2))
```