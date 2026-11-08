```python
import numpy as np
np.set_printoptions(linewidth=180, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def pr(M):
    ev=np.linalg.eigvals(M); return "%+.2f/%.0f"%(ev.real.sum()/np.abs(ev).sum(), np.abs(ev).sum())
print("== LN weight stats")
for l in range(4):
    for n in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        v=W[f'{n}.{l}']; print(f"{n}.{l}: mean {v.mean():.3f} std {v.std():.3f} min {v.min():.3f} max {v.max():.3f}")
for n in ['lnf_w','lnf_b','tok_emb','pos_emb']:
    print(n, n in W)
print("== per head: QK fro, QK pos-ratio, OV fro, OV pos-ratio(trace/sum|ev|), q-bias norm, k-bias norm")
for l in range(4):
    b=W[f'ln1_b.{l}']
    for h in range(6):
        q,k,v,o=head(l,h); s=slice(64*h,64*h+64)
        A=q.T@k; OV=o@v
        qb=W[f'W_q.{l}'][s]@b; kb=W[f'W_k.{l}'][s]@b
        print(f"L{l}H{h}: QK {np.linalg.norm(A):6.1f} {pr(A)} | OV {np.linalg.norm(OV):5.2f} {pr(OV)} | qb {np.linalg.norm(qb):.2f} kb {np.linalg.norm(kb):.2f}")
print("== K-composition matching (query raw residual, key via earlier head OV): ratio/scale")
for l2 in range(1,4):
    for h2 in range(6):
        q2,k2,_,_=head(l2,h2); A=q2.T@k2
        out=[]
        for l1 in range(l2):
            for h1 in range(6):
                _,_,v1,o1=head(l1,h1); out.append(pr(A@o1@v1))
        print(f"L{l2}H{h2}: "+" ".join(out))
```