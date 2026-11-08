```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w=W["ln1_w.0"]
def rsub(M,r):
    U,s,Vt=np.linalg.svd(M); return Vt[:r].T, s
def ov(A,B):  # mean squared cosine of principal angles
    s=np.linalg.svd(A.T@B,compute_uv=False); return s
subs={}
for j in range(6):
    for nm in ["W_q","W_k","W_v"]:
        V,s=rsub(hm(nm,0,j)*w,12); subs[(nm,j)]=V
for j in [0,3,1,5]:
    print(f"L0H{j}: Q-K overlap cos", ov(subs[('W_q',j)],subs[('W_k',j)])[:10])
    print(f"       Q-V", ov(subs[('W_q',j)],subs[('W_v',j)])[:10])
    print(f"       K-V", ov(subs[('W_k',j)],subs[('W_v',j)])[:10])
print("V3 vs V0", ov(subs[('W_v',3)],subs[('W_v',0)])[:10])
print("V3 vs V5", ov(subs[('W_v',3)],subs[('W_v',5)])[:10])
print("K3 vs K0", ov(subs[('W_k',3)],subs[('W_k',0)])[:10])
print("Q3 vs Q0", ov(subs[('W_q',3)],subs[('W_q',0)])[:10])
print("K3 vs V0", ov(subs[('W_k',3)],subs[('W_v',0)])[:10])
# random baseline for 12 vs 12 dims in 384
R1=np.linalg.qr(np.random.randn(384,12))[0];R2=np.linalg.qr(np.random.randn(384,12))[0]
print("random", ov(R1,R2)[:10])
```