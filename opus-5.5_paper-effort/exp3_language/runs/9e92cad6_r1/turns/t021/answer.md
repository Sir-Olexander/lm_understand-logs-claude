```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
rng=np.random.default_rng(0)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
out=[21,39,58,99,184]; keep=np.ones(384,bool); keep[out]=False
def sub(M,k):
    M=M.copy(); M[:,~keep]=0
    _,s,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
def rsub(k):
    X=rng.standard_normal((384,k)); X[~keep]=0; return np.linalg.qr(X)[0]
w0=W["ln1_w.0"]
V=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
FC=W["W_fc.0"]*W["ln2_w.0"]
Pread=np.concatenate([hm("W_q",0,3)*w0,hm("W_k",0,3)*w0,hm("W_k",0,0)*w0,hm("W_v",5,)*w0 if False else hm("W_v",0,5)*w0],0)
for (a,b) in [(24,40),(28,40),(30,50),(36,60)]:
    S1=sub(V,a); S3=sub(FC,b)
    c=np.linalg.svd(S1.T@S3,compute_uv=False)
    cr=np.mean([np.linalg.svd(rsub(a).T@rsub(b),compute_uv=False) for _ in range(5)],0)
    print(f"V top{a} vs FC top{b}:", c, "\n   random:", cr[:10])
# How much of FC0's top-50 is positional? overlap FC top50 with P readers top20
S3=sub(FC,50); SP=sub(Pread,20)
print("FC50 vs P20", np.linalg.svd(SP.T@S3,compute_uv=False))
print("FC sv", np.linalg.svd(np.where(keep[None,:],FC,0),compute_uv=False)[:60])
```