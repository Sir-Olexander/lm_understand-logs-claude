```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
out=[21,39,58,99,184]; keep=np.ones(384,bool); keep[out]=False
def sub(M,k):
    M=M.copy(); M[:,~keep]=0
    _,s,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
w0=W["ln1_w.0"]; w1=W["ln1_w.1"]
V3=hm("W_v",0,3)*w0; V0=hm("W_v",0,0)*w0
L1=np.concatenate([hm(nm,1,j)*w1 for nm in ["W_q","W_k","W_v"] for j in range(6)],0)
FC=W["W_fc.0"]*W["ln2_w.0"]
L3Q=hm("W_q",3,3)*W["ln1_w.3"]
for k in [30,40]:
    S1=sub(np.concatenate([V3,V0],0),k); S2=sub(L1,k+20); S3=sub(FC,k+30); S4=sub(L3Q,k)
    c12=np.linalg.svd(S1.T@S2,compute_uv=False); c13=np.linalg.svd(S1.T@S3,compute_uv=False); c23=np.linalg.svd(S2.T@S3,compute_uv=False)
    c14=np.linalg.svd(S1.T@S4,compute_uv=False)
    print(f"k={k} V0/V3 vs L1:", c12[:36]); print("      V vs FC0:", c13[:36]); print("      L1 vs FC0:", c23[:36]); print("      V vs L3H3q:", c14[:36])
```