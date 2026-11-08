```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
# common ins and outs from sum of group OVs
G=sum(mats(0,j)[3]@mats(0,j)[2] for j in [1,2,4,5])
U,s,Vt=np.linalg.svd(G); outs=U[:,:2]; ins=Vt[:2].T
print('group sum svals',s[:4])
A={}
for j in [1,2,4,5,0,3]:
    Q,K,V,O=mats(0,j); A[j]=outs.T@(O@V)@ins
    print('L0H%d A=\n'%j,A[j])
for l in range(1,4):
    lb=W[f'ln1_b.{l}']
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q,K,V,O=mats(l,j); qb=W[f'W_q.{l}'][s_]@lb; r=K.T@qb/8
        ro=outs.T@r
        print('L%dH%d r_outs'%(l,j),ro, ' per-head ins-row:',' '.join(str(np.round(ro@A[h],2)) for h in [1,2,4,5]))
```