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
OV3=H[0][3][3]@H[0][3][2]
Vt3=np.linalg.svd(OV3)[2]; T=Vt3[:16].T
O3=np.linalg.svd(OV3)[0][:,:16]
Up=np.linalg.svd(W['W_proj.3'],full_matrices=False)[0]
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=H[l][j]; OV=o@v; n=np.linalg.norm(OV)
        print(f"L{l}H{j} OV-in energy: T {np.linalg.norm(OV@T)**2/n**2:.3f} O3 {np.linalg.norm(OV@O3)**2/n**2:.3f} | out in Up21 {np.linalg.norm(Up[:,:21].T@OV)**2/n**2:.3f}  q-in T {np.linalg.norm(q@T)**2/np.linalg.norm(q)**2:.3f} k-in O3 {np.linalg.norm(k@O3)**2/np.linalg.norm(k)**2:.3f} k-in T {np.linalg.norm(k@T)**2/np.linalg.norm(k)**2:.3f}")
print("random 16-dim baseline energy", 16/384)
```