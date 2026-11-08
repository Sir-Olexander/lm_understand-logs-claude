```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q3,K3,V3,O3=mats(0,3); T=sub(O3@V3,25,'in')
cols=[]
for j in range(6):
    Q,K,V,O=mats(0,j); cols+= [sub(Q,10,'in'), sub(K,10,'in')]
A=np.hstack(cols); A=A-T@(T.T@A)
U,s,Vt=np.linalg.svd(A,full_matrices=False)
print('svals of L0 QK-read dirs minus T',s[:40])
Pall=U[:,:30]
f=lambda M,S: np.linalg.norm(M@S)**2/np.linalg.norm(M)**2*384/S.shape[1]
print('L0 fc read Pall %.2f'%f(W['W_fc.0']*W['ln2_w.0'],Pall))
for l in range(1,4):
    print('fc%d read Pall %.2f T %.2f'%(l,f(W[f'W_fc.{l}']*W[f'ln2_w.{l}'],Pall),f(W[f'W_fc.{l}']*W[f'ln2_w.{l}'],T)))
    for j in range(6):
        Q,K,V,O=mats(l,j)
        print('  L%dH%d Pall: Q %.2f K %.2f V %.2f | T: Q %.2f K %.2f V %.2f'%(l,j,f(Q,Pall),f(K,Pall),f(V,Pall),f(Q,T),f(K,T),f(V,T)))
```