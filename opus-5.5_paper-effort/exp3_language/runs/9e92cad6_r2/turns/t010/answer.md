```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
outsub={}
for l in range(3):
    for j in range(6):
        Q,K,V,O=mats(l,j)
        U,s,Vt=np.linalg.svd(O@V)
        e=np.cumsum(s**2)/np.sum(s**2); r=int(np.searchsorted(e,0.8))+1
        outsub[(l,j)]=U[:,:r]
    # MLP output subspace
print({k:v.shape[1] for k,v in outsub.items()})
def frac(M,Usub):  # fraction of row-space energy of M (rows read residual) in Usub, normalized by dim
    return (np.linalg.norm(M@Usub)**2/np.linalg.norm(M)**2)/(Usub.shape[1]/384)
for l2 in range(1,4):
    for j2 in range(6):
        Q,K,V,O=mats(l2,j2)
        rk=[];rq=[];rv=[]
        for (l1,j1),Us in outsub.items():
            if l1>=l2: continue
            rk.append(frac(K,Us)); rq.append(frac(Q,Us)); rv.append(frac(V,Us))
        print(f'L{l2}H{j2} K',np.round(rk,1)); print('     Q',np.round(rq,1)); print('     V',np.round(rv,1))
```