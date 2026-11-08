```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def svdk(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k],S[:k],Vt[:k].T
O3V3=mats(0,3)[3]@mats(0,3)[2]; O4V4=mats(0,4)[3]@mats(0,4)[2]; O0V0=mats(0,0)[3]@mats(0,0)[2]
u3,s3,v3=svdk(O3V3,3); u4,s4,v4=svdk(O4V4,3); u0,s0,v0=svdk(O0V0,3)
print("out overlap o3 vs o4", np.round(np.linalg.svd(u3[:,:2].T@u4[:,:2],compute_uv=False),2))
print("in overlap H3 vs H4", np.round(np.linalg.svd(v3[:,:2].T@v4[:,:2],compute_uv=False),2))
print("in overlap H3 top1 vs H4 plane", np.round(np.linalg.norm(v4[:,:2].T@v3[:,0]),2))
# L1 queries: decompose query side of top QK singular vectors onto o3,o4
for j in range(6):
    Q,K,V,O=mats(1,j); M=Q.T@K
    # restrict key side to o4 plane: Mq = M @ u4 (384x2) -> what query dirs map to it
    Mk=M@u4[:,:2]
    U,S,Vt=np.linalg.svd(Mk)
    qd=U[:,0]
    print(f"L1H{j} key=f-plane sv {np.round(S,2)}; query dir overlap: o4 {np.linalg.norm(u4[:,:2].T@qd):.2f} o3 {np.linalg.norm(u3[:,:2].T@qd):.2f} o0 {np.linalg.norm(u0[:,:2].T@qd):.2f}")
    # 2x2 blocks
    print("    q:o4/k:o4", np.round(u4[:,:2].T@M@u4[:,:2],2).tolist(), " q:o3/k:o4", np.round(u3[:,:2].T@M@u4[:,:2],2).tolist())
```