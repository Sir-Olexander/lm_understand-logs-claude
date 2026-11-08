```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q3,K3,V3,O3=mats(0,3); T=sub(O3@V3,25,'in'); S3=sub(O3@V3,25,'out')
Q0,K0,V0,O0=mats(0,0); S0=sub(O0@V0,10,'out')
G=sum(mats(0,j)[3]@mats(0,j)[2] for j in [1,2,4,5]); outs=sub(G,3,'out')
cols=[]
for j in range(6):
    Q,K,V,O=mats(0,j); cols+= [sub(Q,10,'in'), sub(K,10,'in')]
A=np.hstack(cols); A=A-T@(T.T@A); Pall=np.linalg.svd(A,full_matrices=False)[0][:,:30]
X=np.linalg.qr(np.hstack([Pall,S3,S0,outs]))[0]
Fc=W['W_fc.0']*W['ln2_w.0']
Fr=Fc-(Fc@X)@X.T
U,s,Vt=np.linalg.svd(Fr)
print('fc0 residual svals',s[:50])
Tf=Vt[:30].T
print('overlap of fc0-residual top30 with T25',np.round(np.linalg.svd(Tf.T@T,compute_uv=False),2))
# V of L0H3 and L0H0 restricted to complement of Pall
for j in [3,0]:
    Q,K,V,O=mats(0,j); Vr=V-(V@Pall)@Pall.T
    print('L0H%d V minus P svals'%j,np.linalg.svd(Vr,compute_uv=False)[:35])
```