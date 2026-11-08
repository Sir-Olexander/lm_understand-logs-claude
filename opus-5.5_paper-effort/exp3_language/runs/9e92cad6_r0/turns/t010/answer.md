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
P=np.linalg.qr(np.concatenate([U[:,:16],Vt[:16].T],1))[0]
P=np.linalg.svd(np.concatenate([U[:,:16],Vt[:16].T],1),full_matrices=False)[0][:,:20]
def topin(M,r): return np.linalg.svd(M)[2][:r].T
def ov_sub(a,b,label):
    s=np.linalg.svd(a.T@b,compute_uv=False)
    print(label, np.round(s,2))
for j in range(6):
    q,k,v,o=H[0][j]
    OV=o@v
    r={0:10,1:2,2:2,3:16,4:2,5:3}[j]
    Vin=topin(OV,r)
    ov_sub(Vin,P,f"L0H{j} OV-in top{r} vs P(20) cosines:")
# OV inputs of H0 vs H3
ov_sub(topin(H[0][0][3]@H[0][0][2],10), topin(H[0][3][3]@H[0][3][2],16),"H0 OVin vs H3 OVin")
# also check: W_v.0 overall top input subspace vs P
Wv=W['W_v.0']*W['ln1_w.0']
ov_sub(topin(Wv,20),P,"Wv.0 top20 in vs P")
Wfc=W['W_fc.0']*W['ln2_w.0']
ov_sub(topin(Wfc,20),P,"Wfc.0 top20 in vs P")
```