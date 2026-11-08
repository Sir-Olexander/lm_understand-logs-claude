```python
def topsv(M,k):
    U,s,Vt=np.linalg.svd(M); return U[:,:k],s[:k],Vt[:k].T
outs={};ins={}
for j in range(6):
    Q,K,V,O=H[0][j]
    U,s,Vv=topsv(O@V,3); outs[j]=U; ins[j]=Vv
    Uq,sq,Vq=topsv(Q.T@K,2)
    print(j,"OV sv",s)
for a in [1,2,4,5]:
    print(a,[np.round(np.linalg.svd(outs[a][:,:2].T@outs[b][:,:2],compute_uv=False),2) for b in [1,2,4,5]],
          [np.round(np.linalg.svd(ins[a][:,:2].T@ins[b][:,:2],compute_uv=False),2) for b in [1,2,4,5]])
```