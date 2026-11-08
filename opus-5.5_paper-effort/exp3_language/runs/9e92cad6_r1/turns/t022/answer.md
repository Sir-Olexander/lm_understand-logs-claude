```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
rng=np.random.default_rng(1)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
out=[21,39,58,99,184]; keep=np.ones(384,bool); keep[out]=False
def sub(M,k):
    M=M.copy(); M[:,~keep]=0
    _,s,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
w0=W["ln1_w.0"]
V=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
FC=W["W_fc.0"]*W["ln2_w.0"]
S1=sub(V,30)
for b in [40,60,80,100,130]:
    S3=sub(FC,b); c=np.linalg.svd(S1.T@S3,compute_uv=False)
    # conditional random: keep first 22 shared dims, random rest
    print(f"FC top{b}:", c[18:30])
for a in [20,24,28,32]:
    S1=sub(V,a); S3=sub(FC,60); c=np.linalg.svd(S1.T@S3,compute_uv=False)
    print(f"V top{a} vs FC60:", c[16:])
# single heads
for j in [3,0]:
    S1=sub(hm("W_v",0,j)*w0,30); S3=sub(FC,50); print(f"V{j} top30 vs FC50", np.linalg.svd(S1.T@S3,compute_uv=False)[14:30])
```