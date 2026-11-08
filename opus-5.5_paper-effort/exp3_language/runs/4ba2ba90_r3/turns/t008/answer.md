```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,k): U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
q,k,v,o=head(0,2); A=q.T@k
U,s,Vt=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U[:,:18],Vt[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v); T=Vt2[:24].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
src={'tok':T,'pos':Ppos,'prevOut':orth(o@v,20)}
for h in [0,1,4,5]:
    q,k,v,o=head(0,h); src[f'b{h}out']=orth(o@v,3)
q,k,v,o=head(0,3); src['h3out']=orth(o@v,5)
src['mlp0']=orth(W['W_proj.0'],40)
names=list(src)
print("sources:",names)
for n1 in names:
    print(n1,[round(float(np.linalg.svd(src[n1].T@src[n2],compute_uv=False)[0]),2) for n2 in names])
def posr(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/(np.abs(ev).sum()+1e-12)
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=head(l,h); A=q.T@k
        nq={n:np.linalg.norm(src[n].T@q.T)/np.sqrt(src[n].shape[1]) for n in names}
        nk={n:np.linalg.norm(k@src[n])/np.sqrt(src[n].shape[1]) for n in names}
        print(f"L{l}H{h} q-read:"," ".join(f"{n}:{nq[n]:.2f}" for n in names))
        print(f"      k-read:"," ".join(f"{n}:{nk[n]:.2f}" for n in names), " rand:%.2f"%(np.linalg.norm(k)/np.sqrt(384)))
```