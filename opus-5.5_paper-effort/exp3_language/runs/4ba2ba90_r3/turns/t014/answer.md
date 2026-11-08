```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def hb(l,h):
    s=slice(64*h,64*h+64); b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}'][s]@b, W[f'W_k.{l}'][s]@b
OVb=sum(head(0,h)[3]@head(0,h)[2] for h in [0,1,3,4,5])
Ub,sb,Vtb=np.linalg.svd(OVb); B2=Ub[:,:2]; B3=Ub[:,:3]
# per-broad-head output in B2 plane: their OV maps -> 2x? ; show each head's OV projected
for h in [0,1,3,4,5]:
    q,k,v,o=head(0,h); M=B3.T@o@v
    print(f"L0H{h} OV in b-plane: norm {np.linalg.norm(M):.2f}, per b-dir",np.linalg.norm(M,axis=1))
# inputs of the b-plane: which Ppos directions feed it; relation to L0H2 planes
q,k,v,o=head(0,2); A0=q.T@k
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=head(l,h); qb,kb=hb(l,h); A=q.T@k
        M=B2.T@A@B2; a,b_,c,d=M[0,0],M[0,1],M[1,0],M[1,1]
        rot=np.hypot(a+d,c-b_)/2; ref=np.hypot(a-d,b_+c)/2; ang=np.arctan2(c-b_,a+d)/np.pi
        gk=B3.T@(k.T@qb)
        print(f"L{l}H{h}: b-plane QK rot {rot:.2f} angle {ang:+.2f}pi refl {ref:.2f} | qbias-pref on b (3d): {np.round(gk,2)}")
```