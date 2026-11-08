```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=mats(0,3); U3,s3,Vt3=np.linalg.svd(O3@V3)
Fc=W['W_fc.0']*W['ln2_w.0']
# fc0 rows: how much in T(k) for k=10..40
for k in [10,15,20,25,30,40]:
    T=Vt3[:k].T; S=U3[:,:k]
    print(k,'fc0 energy frac in T %.3f, in S %.3f (rand %.3f)'%(np.linalg.norm(Fc@T)**2/np.linalg.norm(Fc)**2, np.linalg.norm(Fc@S)**2/np.linalg.norm(Fc)**2, k/384))
# spectrum of fc0 restricted to T30 and S30
T=Vt3[:30].T; S=U3[:,:30]
print('fc0@T svals',np.linalg.svd(Fc@T,compute_uv=False))
print('fc0@S svals',np.linalg.svd(Fc@S,compute_uv=False))
```