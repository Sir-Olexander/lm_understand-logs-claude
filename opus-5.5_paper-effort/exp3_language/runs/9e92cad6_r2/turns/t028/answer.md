```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
f=lambda M,S: np.linalg.norm(M@S)**2/np.linalg.norm(M)**2*384/S.shape[1]
for l in range(3):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False); print('W_proj.%d svals'%l, s[:12], s[30:80:10])
M0=sub(W['W_proj.0'],40,'out'); M1=sub(W['W_proj.1'],40,'out'); M2=sub(W['W_proj.2'],40,'out')
Q3,K3,V3,O3=mats(0,3); T=sub(O3@V3,25,'in')
print('overlap M0 with T', np.round(np.linalg.svd(M0.T@T,compute_uv=False)[:10],2))
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j)
        row='L%dH%d MLP0out: Q %.2f K %.2f V %.2f'%(l,j,f(Q,M0),f(K,M0),f(V,M0))
        if l>=2: row+=' | MLP1out: Q %.2f K %.2f V %.2f'%(f(Q,M1),f(K,M1),f(V,M1))
        if l>=3: row+=' | MLP2out: Q %.2f K %.2f V %.2f'%(f(Q,M2),f(K,M2),f(V,M2))
        print(row)
```