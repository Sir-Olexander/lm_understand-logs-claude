```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def headov(l,j):
    s=slice(64*j,64*j+64); g=W64[f'ln1_w.{l}']
    return W64[f'W_o.{l}'][:,s]@(W64[f'W_v.{l}'][s]*g)
Up,sp,_=np.linalg.svd(W64['W_proj.3'])
for j in range(6):
    ovm=headov(3,j)
    r=np.linalg.norm(Up[:,:34].T@ovm,axis=1)/np.linalg.norm(ovm)*np.sqrt(384)
    print("L3H",j, r)
# MLP2 and attention 2 outputs too
r=np.linalg.norm(Up[:,:34].T@W64['W_proj.2'],axis=1)/np.linalg.norm(W64['W_proj.2'])*np.sqrt(384); print("Wproj2",r)
print("sp",sp[:34])
```