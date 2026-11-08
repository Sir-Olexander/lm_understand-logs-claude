```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
o4=topleft(mats(0,4)[3]@mats(0,4)[2],3)
o0=topleft(mats(0,0)[3]@mats(0,0)[2],3)
print("o0.4 sv", np.round(np.linalg.svd(mats(0,4)[3]@mats(0,4)[2],compute_uv=False)[:4],2))
for l in range(1,4):
  for j in range(6):
    Q,K,V,O=mats(l,j)
    # SVD of QK: top singular pairs, check where q-side (left) and k-side (right) live
    M=Q.T@K
    U,S,Vt=np.linalg.svd(M)
    out=[]
    for r in range(4):
        out.append(f"s{S[r]:.2f} q|o4 {np.linalg.norm(o4[:,:2].T@U[:,r]):.2f} k|o4 {np.linalg.norm(o4[:,:2].T@Vt[r]):.2f}")
    print(f"L{l}H{j}", " ; ".join(out))
```