```python
Up,Sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Uo,So,_=np.linalg.svd(W['W_o.3'],full_matrices=False)
for r in [24,26,28,30,34,40]:
    c=np.linalg.svd(Up[:,:r].T@Uo[:,:r],compute_uv=False)
    print(r, np.round(c,2))
# per-head output spaces of L3 (OV left sv, weighted) vs Up
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H3=heads(3)
for j in range(6):
    q,k,v,o=H3[j]; OV=o@v
    # fraction of OV output energy in top-r of Up
    for r in [21,24,28,32]:
        pass
    print(j,[round(np.linalg.norm(Up[:,:r].T@OV)**2/np.linalg.norm(OV)**2,3) for r in [21,24,26,28,30,32,40]])
print("W_proj.2 energy in Up:",[round(np.linalg.norm(Up[:,:r].T@W['W_proj.2'])**2/np.linalg.norm(W['W_proj.2'])**2,3) for r in [21,24,26,28,30,32,40]])
print("W_proj.3 energy:",[round((Sp[:r]**2).sum()/(Sp**2).sum(),3) for r in [21,24,26,28,30,32,40]])
```