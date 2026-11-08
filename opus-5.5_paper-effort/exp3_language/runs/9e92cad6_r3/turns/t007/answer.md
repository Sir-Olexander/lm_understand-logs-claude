```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return U[:,:k] if side=='L' else Vt[:k].T
def overlap(A,B):  # mean squared cosine of principal angles
    s=np.linalg.svd(A.T@B,compute_uv=False); return np.round(s,2)
k=12
for j in [0,3]:
    Q,K,V,O=H[0][j]
    qs=sub(Q.T@K,k,'L'); ks=sub(Q.T@K,k,'R'); vs=sub(O@V,k,'R')
    print("head",j)
    print(" q vs k", overlap(qs,ks))
    print(" q vs ov-in", overlap(qs,vs))
    print(" k vs ov-in", overlap(ks,vs))
Q,K,V,O=H[0][5]; f2=sub(O@V,2,'R')
for j in [0,3]:
    Q,K,V,O=H[0][j]
    print(j,"2d-feature vs q,k,ovin", overlap(f2,sub(Q.T@K,k,'L')),overlap(f2,sub(Q.T@K,k,'R')),overlap(f2,sub(O@V,k,'R')))
# compare H0 and H3 subspaces
Q0,K0,V0,O0=H[0][0];Q3,K3,V3,O3=H[0][3]
print("H0 k vs H3 k",overlap(sub(Q0.T@K0,k,'R'),sub(Q3.T@K3,k,'R')))
print("H0 q vs H3 q",overlap(sub(Q0.T@K0,k,'L'),sub(Q3.T@K3,k,'L')))
print("H0 ovin vs H3 ovin",overlap(sub(O0@V0,k,'R'),sub(O3@V3,k,'R')))
print("H0 ovout vs H3 ovout",overlap(sub(O0@V0,k,'L'),sub(O3@V3,k,'L')))
```