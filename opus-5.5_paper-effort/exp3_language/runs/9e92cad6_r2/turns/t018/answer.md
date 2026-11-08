```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q3,K3,V3,O3=mats(0,3); OV3=O3@V3
U,s,Vt=np.linalg.svd(OV3); T=Vt[:15].T; S=U[:,:15]
print('OV3 svals',s[:16])
# random 15-dim baseline
rng=np.random.default_rng(1)
R=np.linalg.qr(rng.normal(size=(384,15)))[0]
for (l,j) in [(3,3),(3,0),(3,4),(2,4),(1,0),(1,2)]:
    Q,K,V,O=mats(l,j); QK=Q.T@K/8
    for qn,Aq in [('T',T),('S',S),('R',R)]:
        for kn,Ak in [('T',T),('S',S),('R',R)]:
            B=Aq.T@QK@Ak
            ev=np.linalg.eigvals(B); 
            print('L%dH%d q%s k%s |B| %.3f pos %.2f trace %.3f'%(l,j,qn,kn,np.linalg.norm(B),ev.real.sum()/np.abs(ev).sum(),np.trace(B)),end=' ; ')
        print()
```