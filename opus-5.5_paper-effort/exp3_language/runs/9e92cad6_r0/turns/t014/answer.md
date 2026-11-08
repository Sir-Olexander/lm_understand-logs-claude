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
q,k,v,o=H[3][3]
M=(q.T@k)@OV3
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-ev.real)]
print("eig of QK(L3H3) OV(L0H3):", np.round(ev[:40],3))
# symmetric part analysis in original coords: M_sym eigen
Ms=(M+M.T)/2
es=np.linalg.eigvalsh(Ms)[::-1]
print("sym eig top", np.round(es[:40],3)); print("sym eig bottom", np.round(es[-10:],3))
print("sv of M", np.round(np.linalg.svd(M,compute_uv=False)[:40],3))
for nm,(qq,kk,vv,oo) in [("L3H3",H[3][3]),("L3H4",H[3][4]),("L3H0",H[3][0])]:
    print(nm,"OV sv",np.round(np.linalg.svd(oo@vv,compute_uv=False)[:45],2))
```