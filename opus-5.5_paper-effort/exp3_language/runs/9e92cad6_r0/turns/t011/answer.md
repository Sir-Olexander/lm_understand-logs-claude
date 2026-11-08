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
q,k,v,o=H[0][3]
U,S,Vt=np.linalg.svd(q.T@k)
P=np.linalg.svd(np.concatenate([U[:,:16],Vt[:16].T],1),full_matrices=False)[0][:,:20]
U3,S3,Vt3=np.linalg.svd(o@v); T=Vt3[:16].T
U0,S0,Vt0=np.linalg.svd(H[0][0][3]@H[0][0][2]); T=np.linalg.svd(np.concatenate([Vt3[:16].T,Vt0[:10].T],1),full_matrices=False)[0][:,:20]
print("P vs T overlap", np.round(np.linalg.svd(P.T@T,compute_uv=False),2))
for j in range(6):
    q,k,v,o=H[0][j]
    Uq,Sq,Vtq=np.linalg.svd(q.T@k)
    for r in range(2):
        u=Uq[:,r]; kk=Vtq[r]
        print(f"L0H{j} sv{r} {Sq[r]:.2f}: query in P {np.linalg.norm(P.T@u):.2f} in T {np.linalg.norm(T.T@u):.2f} | key in P {np.linalg.norm(P.T@kk):.2f} in T {np.linalg.norm(T.T@kk):.2f}")
# bias contributions
b=W['ln1_b.0']
for j in range(6):
    s=slice(j*64,(j+1)*64)
    bq=W['W_q.0'][s]@b; bk=W['W_k.0'][s]@b
    q,k,v,o=H[0][j]
    kb = bq@k  # key-only preference vector (384) from query bias
    print(f"H{j} |bq| {np.linalg.norm(bq):.3f} key-pref vec norm {np.linalg.norm(kb):.3f}  in P {np.linalg.norm(P.T@kb)/np.linalg.norm(kb):.2f} in T {np.linalg.norm(T.T@kb)/np.linalg.norm(kb):.2f}")
```