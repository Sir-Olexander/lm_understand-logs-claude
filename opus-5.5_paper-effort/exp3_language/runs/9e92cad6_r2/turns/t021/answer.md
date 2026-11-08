```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=mats(0,3); U,s,Vt=np.linalg.svd(O3@V3); T=Vt[:20].T
G=sum(mats(0,j)[3]@mats(0,j)[2] for j in [1,2,4,5]); Ug,sg,Vg=np.linalg.svd(G); outs=Ug[:,:2]; ins=Vg[:2].T
vecs=[];names=[]
for j in [1,2,4,5]:
    Q,K,V,O=mats(0,j); QK=Q.T@K/8; Uq,sv,Vk=np.linalg.svd(QK); v=Vk[0]
    vt=v-ins@(ins.T@v); vecs.append(vt/np.linalg.norm(vt)); names.append('g%d'%j)
for l in range(1,4):
    lb=W[f'ln1_b.{l}']
    for j in range(6):
        s_=slice(64*j,64*j+64); Q,K,V,O=mats(l,j); qb=W[f'W_q.{l}'][s_]@lb; r=K.T@qb/8
        r=r-outs@(outs.T@r)
        rt=T@(T.T@r); vecs.append(rt/np.linalg.norm(rt)); names.append('L%dH%d'%(l,j))
Vm=np.array(vecs); C=Vm@Vm.T
print('      '+' '.join('%5s'%n for n in names))
for i,n in enumerate(names): print('%5s '%n+' '.join('%5.2f'%c for c in C[i]))
```