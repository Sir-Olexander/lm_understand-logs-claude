```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w0=W["ln1_w.0"]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
U,s,Vt=np.linalg.svd(np.concatenate([hm("W_v",0,3)*w0, hm("W_v",0,0)*w0],0),full_matrices=False)
T=Vt[:22].T
O3=OV(0,3); O0=OV(0,0)
def stats(A):
    ev=np.linalg.eigvals(A); m=np.abs(ev); a=np.angle(ev)
    pos=m[np.abs(a)<np.radians(30)].sum(); neg=m[np.abs(a)>np.radians(150)].sum()
    sym=(A+A.T)/2; es=np.linalg.eigvalsh(sym)
    return f"tr {np.trace(A):7.2f} pos {pos:6.2f} neg {neg:6.2f} tot {m.sum():6.2f} symeig+ {es[es>0].sum():6.2f} symeig- {es[es<0].sum():6.2f}"
for l in range(1,4):
    w=W[f"ln1_w.{l}"]
    for j in range(6):
        Q=hm("W_q",l,j)*w; K=hm("W_k",l,j)*w
        A_ind=T.T@Q.T@K@O3@T     # query token t (T) vs key prev-token via L0H3
        A_rev=T.T@O3.T@Q.T@K@T   # query prev-token (via L0H3) vs key token
        A_same=T.T@Q.T@K@T       # query token vs key token
        print(f"L{l}H{j} ind: {stats(A_ind)}")
        print(f"      rev: {stats(A_rev)}")
        print(f"      same:{stats(A_same)}")
```