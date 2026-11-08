```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(3)]
Up,sp,_=np.linalg.svd(W64['W_proj.3'])
L=Up[:,:30]
for j in range(6):
    Q,K,V,O=H[0][j]; OV=O@V
    s=np.linalg.svd(OV,compute_uv=False)
    sl=np.linalg.svd(L[:,:28].T@OV,compute_uv=False)
    print(j, s[:36])
    print("   frac of OV in Wproj3 top28 subspace: %.2f"%(np.linalg.norm(L[:,:28].T@OV)/np.linalg.norm(OV)), " top21 %.2f"%(np.linalg.norm(L[:,:21].T@OV)/np.linalg.norm(OV)))
# random baseline fraction
print("baseline 28/384 sqrt", np.sqrt(28/384))
```