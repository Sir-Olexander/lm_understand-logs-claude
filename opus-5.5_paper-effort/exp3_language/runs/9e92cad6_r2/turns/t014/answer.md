```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
f=lambda M,S: np.linalg.norm(M@S)**2/np.linalg.norm(M)**2*384/S.shape[1]
Q3,K3,V3,O3=mats(0,3); Q0,K0,V0,O0=mats(0,0)
S3out=sub(O3@V3,15,'out'); S3in=sub(O3@V3,15,'in')
S0out=sub(O0@V0,8,'out'); S0in=sub(O0@V0,8,'in')
P=np.linalg.qr(np.hstack([sub(Q3,20,'in'),sub(K3,20,'in')]))[0][:,:30]
for l in range(4):
    Wfc=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print('fc%d: L0H3out %.1f L0H3in(tok) %.1f L0H0out %.1f L0H0in %.1f P(L0H3 qk) %.1f'%(l,f(Wfc,S3out),f(Wfc,S3in),f(Wfc,S0out),f(Wfc,S0in),f(Wfc,P)))
for l in range(4):
    for j in range(6):
        Q,K,V,O=mats(l,j)
        print('L%dH%d tok(L0H3in): Q %.1f K %.1f V %.1f | P: Q %.1f K %.1f V %.1f | L0H3out: Q %.1f K %.1f V %.1f'%(l,j,f(Q,S3in),f(K,S3in),f(V,S3in),f(Q,P),f(K,P),f(V,P),f(Q,S3out),f(K,S3out),f(V,S3out)))
# overlap between subspaces
print('S3in vs P', np.round(np.linalg.svd(S3in.T@P,compute_uv=False),2))
print('S3out vs S3in', np.round(np.linalg.svd(S3out.T@S3in,compute_uv=False),2))
print('S0in vs S3in', np.round(np.linalg.svd(S0in.T@S3in,compute_uv=False),2))
```