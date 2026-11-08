```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def hb(l,h):
    s=slice(64*h,64*h+64); b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}'][s]@b, W[f'W_k.{l}'][s]@b, W[f'W_v.{l}'][s]@b
q,k,v,o=head(0,2); A=q.T@k
U0,s0,Vt0=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v); T=Vt2[:28].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
vecs={}
for h in range(6):
    q,k,v,o=head(0,h); qb,kb,vb=hb(0,h); A=q.T@k
    U,s,Vt=np.linalg.svd(A)
    g=k.T@qb   # query-bias-driven key preference (x̂ space)
    vecs[h]=(U[:,0],Vt[0],g)
    print(f"L0H{h}: s1 {s[0]:.2f} |g| {np.linalg.norm(g):.2f} | u1: pos {np.linalg.norm(Ppos.T@U[:,0]):.2f} tok {np.linalg.norm(T.T@U[:,0]):.2f} | v1: pos {np.linalg.norm(Ppos.T@Vt[0]):.2f} tok {np.linalg.norm(T.T@Vt[0]):.2f} | g: pos {np.linalg.norm(Ppos.T@g)/np.linalg.norm(g):.2f} tok {np.linalg.norm(T.T@g)/np.linalg.norm(g):.2f}")
print("cosines between key-side top vectors v1 (abs):")
for a in range(6): print(a,[round(abs(float(vecs[a][1]@vecs[b][1])),2) for b in range(6)])
print("cosines between query-side top vectors u1 (abs):")
for a in range(6): print(a,[round(abs(float(vecs[a][0]@vecs[b][0])),2) for b in range(6)])
print("signed cos of T-projection of v1 between heads:")
for a in range(6):
    pa=T.T@vecs[a][1]; 
    print(a,[round(float(pa@(T.T@vecs[b][1])/np.linalg.norm(pa)/np.linalg.norm(T.T@vecs[b][1])),2) for b in range(6)])
# ln1_w.0 / ln1_b.0 in T and Ppos
w=W['ln1_w.0']; b=W['ln1_b.0']
print("ln1_b.0 norm %.2f, on pos %.2f, on tok %.2f"%(np.linalg.norm(b),np.linalg.norm(Ppos.T@b),np.linalg.norm(T.T@b)))
```