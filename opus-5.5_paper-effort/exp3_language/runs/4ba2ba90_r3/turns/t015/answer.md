```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def hb(l,h):
    s=slice(64*h,64*h+64); b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}'][s]@b, W[f'W_k.{l}'][s]@b
def orth(M,k): U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
def posr(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/(np.abs(ev).sum()+1e-12)
OVb=sum(head(0,h)[3]@head(0,h)[2] for h in [0,1,3,4,5]); Bs=orth(OVb,3)
q,k,v,o=head(0,2); A0=q.T@k; OVp=o@v
U0,s0,Vt0=np.linalg.svd(A0); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(OVp); T=Vt2[:28].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
P0=orth(OVp,24)
print("cols: tt(pos) | t->prevOut(induct,pos) | prevOut->t | pp->pp(pos) | pos-pos(pos) [plane angle avg] | qb.k on tok,pos,prevOut,b")
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=head(l,h); qb,kb=hb(l,h); A=q.T@k
        tt=T.T@A@T; tp=T.T@A@P0; pt=P0.T@A@T; pp=P0.T@A@P0; ps=Ppos.T@A@Ppos
        g=k.T@qb
        print(f"L{l}H{h}: tt {np.linalg.norm(tt):4.2f}({posr(tt):+.2f}) | ind {np.linalg.norm(T.T@A@OVp@T):4.2f}({posr(T.T@A@OVp@T):+.2f}) | prv->t {np.linalg.norm(pt):4.2f} | pp {np.linalg.norm(pp):4.2f}({posr(pp):+.2f}) | pos {np.linalg.norm(ps):4.2f}({posr(ps):+.2f}) | g tok {np.linalg.norm(T.T@g):5.2f} pos {np.linalg.norm(Ppos.T@g):5.2f} prv {np.linalg.norm(P0.T@g):5.2f} b {np.linalg.norm(Bs.T@g):5.2f}")
```