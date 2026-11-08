```python
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
def orth(A,r=None):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return U[:,:r] if r else U
OV=[H[0][j][3]@H[0][j][2] for j in range(6)]
q,k,v,o=H[0][3]; U,S,Vt=np.linalg.svd(q.T@k)
P=orth(np.concatenate([U[:,:16],Vt[:16].T],1),20)
T=np.linalg.svd(OV[3])[2][:16].T
F=orth(OV[5],2)
O3=orth(OV[3],16)
O0=orth(OV[0],10)
M0=orth(W['W_proj.0'],40)
bases={'T':T,'P':P,'F':F,'O3':O3,'O0':O0,'M0':M0}
names=list(bases)
print("pairwise overlaps (largest cos):")
for a in names:
    print(a,[round(np.linalg.svd(bases[a].T@bases[b],compute_uv=False)[0],2) for b in names])
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=H[l][j]; QK=q.T@k; n=np.linalg.norm(QK)
        s=[]
        for a in names:
            s.append(" ".join(f"{np.linalg.norm(bases[a].T@QK@bases[b])/n:.2f}" for b in names))
        print(f"L{l}H{j} rows=query {names} cols=key: "+" | ".join(s))
```